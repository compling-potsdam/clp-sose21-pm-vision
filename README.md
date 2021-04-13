# A MapWorld Avatar Game (SoSe21 PM Vision)

# Deployment

## A. Run everything on localhost

### Prepare servers and data

1. Checkout `slurk` and run slurk `local_run`. This will start slurk on `localhost:5000`. This will also create the
   default admin token.
1. Download and unpack the ADE20K dataset. Go into the images training directory and start a http server as the server
   serving the images. You can use `python -m http.server 8000` for this.
1. Checkout `clp-sose21-pm-vision` and run the `game_setup_cli`. By default, the script expects slurk to run
   on `localhost:5000`. If slurk runs on another machine, then you must provide the `--slurk_host` and `--slurk_port`
   options. If you do not use the default admin token, then you must use the `--token` option to provide the admin
   token. The script will create the game room, task and layout using the default admin token via slurks REST API. This
   will also create three tokens: one for the game master, player and avatar. See the console output for these tokens.

### Prepare clients and bots

1. Run the `game_master_cli --token <master-token>`. This will connect the game master with slurk. By default, this will
   create image-urls that point to `localhost:8000`.
1. Run a (private-mode) browser and go to `localhost:5000` and login as `Player` using the `<player-token>`.
1. Run a different (private-mode) browser and go to `localhost:5000` and login as `Avatar` using the `<avatar-token>`.
   If the avatar is a bot, then the bot will have to use the token, when the connection is about to be established, just
   like with the game master.

## B. Run everything with ngrok (temporarily externally available)

First, do everything as in *Run everything on localhost: Prepare servers and data*.

### Prepare ngrok

1. Externalize the slurk server using `ngrok http 5000` which will give you something like `<slurk-hash>.ngrok.io`
1. Externalize he timage server using `ngrok http 8000` which will give you something like `<image-hash>.ngrok.io`

### Prepare clients and bots

1. Run the `game_master_cli --token <master-token> --image_server_host <image-hash>.ngrok.io --image_server_port 80`.
   This will connect the game master with slurk. This will create image-urls that point to `<image-hash>.ngrok.io:80`.
   If you run the game master on the same host as slurk, then the game master will automatically connect
   to `localhost:5000`. If you run the game master from another machine, then you probably want to also provide
   the `--slurk_host <slurk-hash>.ngrok.io` and `--slurk_port 80` options.
1. Run a (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Player` using the player token. You
   might have to wait a minute until you can also connect as the second player.
1. Run a different (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Avatar` using the avatar
   token. If the avatar is a bot, then the bot will have to use the token, when the connection is about to be
   established, just like with the game master.

## Installation

Run `update.sh` to install the project on a machine. This will install the python project as an egg
into `$HOME/.local/lib/pythonX.Y/site-packages`. Add the install directory to your python path to make the app
available:

`export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/pythonX.Y/site-packages`

Notice: Use the python version X.Y of your choice. Preferebly add this export also to your `.bashrc`.

You can also simply perform `python setup.py install`. This will install the app into the currently activate python
environment.


