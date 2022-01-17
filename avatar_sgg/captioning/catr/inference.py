import torch
import torch.nn.functional

from transformers import BertTokenizer
from PIL import Image
from avatar_sgg.captioning.catr.hubconf import v3
from avatar_sgg.captioning.catr.datasets import coco
from avatar_sgg.captioning.catr.configuration import Config
from avatar_sgg.config.util import get_config
import numpy as np
import os



class CATRInference():
    def __init__(self):
        self.config = Config()
        # set local file to True if you have connection issues...
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
        self.max_length = self.config.max_position_embeddings
        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)
        self.model = v3(pretrained=True)#torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        self.model.eval()
        catr_config = get_config()["captioning"]["catr"]
        self.cuda_device = catr_config["cuda_device"]
        self.beam_size = catr_config["beam_size"]

        if type(self.cuda_device) is str and self.cuda_device.startswith("cuda"):
            print("Use CATR Model with GPU", self.cuda_device)
            self.model.cuda(self.cuda_device)
        else:
            print("Use CATR Model with CPU")

    def create_caption_and_mask(self):



        self.caption_template = torch.zeros((1, self.max_length), dtype=torch.long)
        self.mask_template = torch.ones((1, self.max_length), dtype=torch.bool)

        self.caption_template[:, 0] = self.start_token
        self.mask_template[:, 0] = False

        return self.caption_template, self.mask_template
    @torch.no_grad()
    def infer_beam(self, image_path):
        """
        Beam Search still broken. Does not deliver better bleu score than greedy search
        :param image_path:
        :return:
        """
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        beam_size = self.beam_size
        caption, cap_mask = self.create_caption_and_mask()

        if self.cuda_device.startswith("cuda"):
            image = image.cuda(self.cuda_device)
            caption = caption.cuda(self.cuda_device)
            cap_mask = cap_mask.cuda(self.cuda_device)
        src, mask, pos = self.model.init_sample(image)

        predictions = self.model.infer(src, mask, pos, caption, cap_mask) #self.model(image, caption, cap_mask)
        predictions = torch.nn.functional.log_softmax(predictions[:, 0, :], dim=-1)#predictions[:, 0, :]#torch.nn.functional.log_softmax(predictions[:, 0, :])
        previous_log_prob, candidate_indices = torch.topk(predictions, beam_size)
        preds = {i: np.zeros(self.max_length, dtype=int) for i in range(beam_size)}
        for i in range(beam_size):
            preds[i][0] = candidate_indices[0][i]
        # Copy entries a number of time equal to the beam size (the number of alternative paths)
        # 1 means the dimensions stay untouched
        #image = image.repeat(beam_size, 1, 1, 1)
        caption = caption.repeat(beam_size, 1)
        cap_mask = cap_mask.repeat(beam_size, 1)
        src = src.repeat(beam_size,1 ,1 ,1)
        mask = mask.repeat(beam_size,1 ,1)
        pos[0] = pos[0].repeat(beam_size,1, 1,1)
        candidates = []
        caption[:, 1] = candidate_indices
        cap_mask[:, 1] = False
        for step in range(1, self.max_length - 1):
            predictions = self.model.infer(src, mask, pos, caption, cap_mask) #self.model(image, caption, cap_mask)
            predictions = torch.nn.functional.log_softmax(predictions[:, step, :], dim=-1)#predictions[:, step, :]
            candidates_log_prob, candidate_indices = torch.topk(predictions, beam_size)
            candidates_log_prob = torch.reshape(candidates_log_prob + previous_log_prob, (-1,))
            candidate_indices = torch.reshape(candidate_indices, (-1,))
            current_top_candidates, current_top_candidates_idx = torch.topk(candidates_log_prob, k=beam_size)

            # Do the mapping best candidate and "source" of the best candidates
            k_idx = torch.index_select(candidate_indices, dim=0, index=current_top_candidates_idx)
            prev_idx = torch.floor(current_top_candidates_idx / beam_size).to(torch.int32)

            previous_log_prob = torch.unsqueeze(current_top_candidates, dim=1)
            np_prev_idx = prev_idx.cpu().numpy()
            # Overwrite the previous predictions due to the new best candidates
            temp = caption.clone()
            for i in range(prev_idx.shape[0]):
                temp[i][:step + 1] = caption[np_prev_idx[i]][:step + 1]
            caption = temp
            preds = {i: preds[np_prev_idx[i]].copy() for i in range(prev_idx.shape[0])}
            caption[:, step + 1] = k_idx
            cap_mask[:, step + 1] = False
            stop_idx = []
            for i in range(k_idx.shape[0]):
                preds[i][step] = k_idx[i]
                if k_idx[i] == self.end_token:
                    stop_idx.append(i)

            # remove all finished captions and adjust all tensors accordingly...
            if len(stop_idx):
                for i in reversed(sorted(stop_idx)):
                    candidate = preds.pop(i)
                    loss = current_top_candidates[i]
                    length = np.where(candidate == self.end_token)[0]
                    normalized_loss = loss / float(length)
                    candidates.append((candidate, normalized_loss))
                beam_size = beam_size - len(stop_idx)
                if beam_size > 0:
                    left_idx = torch.LongTensor([i for i in range(k_idx.shape[0]) if i not in stop_idx])

                    if self.cuda_device.startswith("cuda"):
                        left_idx = left_idx.cuda(self.cuda_device)
                    # current_top_candidates = torch.IntTensor(
                    #     [current_top_candidates[i] for i in range(current_top_candidates.shape[0]) if
                    #      i not in stop_idx])
                    caption = torch.index_select(caption, dim=0, index=left_idx)
                    cap_mask = torch.index_select(cap_mask, dim=0, index=left_idx)
                    #image = torch.index_select(image, dim=0, index=left_idx)
                    previous_log_prob = torch.index_select(previous_log_prob, dim=0, index=left_idx)
                    src = torch.index_select(src, dim=0, index=left_idx)
                    mask = torch.index_select(mask, dim=0, index=left_idx)
                    pos[0] = torch.index_select(pos[0], dim=0, index=left_idx)
                    # now that the finished sentences have been removed, we need to update the predictions dict accordingly
                    for i, key in enumerate(sorted(preds.keys())):
                        preds[i] = preds.pop(key)
                else:
                    break  # No sequences unfinished



        if len(candidates) > 0:
            result, _ = max(candidates, key=lambda c: c[1])
        else:
            result = preds[0]

        output = self.tokenizer.decode(result, skip_special_tokens=True)
        return output


    @torch.no_grad()
    def infer(self, image_path):
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)

        caption, cap_mask = self.create_caption_and_mask()
        if self.cuda_device.startswith("cuda"):
            image = image.cuda(self.cuda_device)
            caption = caption.cuda(self.cuda_device)
            cap_mask = cap_mask.cuda(self.cuda_device)

        src, mask, pos = self.model.init_sample(image)
        #model.eval()
        for i in range(self.max_length - 1):
            predictions = self.model.infer(src, mask, pos, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            if predicted_id[0] == self.end_token:
                #return caption
                break
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
        output = self.tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
        return output


if __name__ == "__main__":
    ade20k = get_config()["ade20k"]["root_dir"]
    image_path = os.path.join(ade20k, "images/training/u/utility_room/ADE_train_00019432.jpg")
    catr = CATRInference()
    output = catr.infer(image_path)
    #result = catr.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
