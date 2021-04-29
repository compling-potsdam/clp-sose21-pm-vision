# A MapWorld Avatar Game (SoSe21 PM Vision)

# Installation

You can install the scripts to be available from the shell (where the python environment is accessible).

For this simply checkout this repository and perform `python setup.py install` from within the root directory. This will
install the app into the currently activate python environment. After installation, you can use the `game-setup`
, `game-master` and `game-avatar` cli commands (where the python environment is accessible).

**Installation for developers on remote machines**
Run `update.sh` to install the project on a machine. This shell script simply pulls the latest changes and performs the
install from above. As a result, the script will install the python project as an egg
into `$HOME/.local/lib/pythonX.Y/site-packages`.

You have to add the install directory to your python path to make the app
available `export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/pythonX.Y/site-packages`

Notice: Use the python version X.Y of your choice. Preferebly add this export also to your `.bashrc`.

# Deployment

## A. Run everything on localhost

### Prepare servers and data

#### 1. Start slurk

Checkout `slurk` and run slurk `local_run`. This will start slurk on `localhost:5000`. This will also create the default
admin token.

#### 2. Download and expose the dataset

Download and unpack the ADE20K dataset. Go into the images training directory and start a http server as the server
serving the images. You can use `python -m http.server 8000` for this.

#### 3. Create the slurk game room and player tokens

Checkout `clp-sose21-pm-vision` and run the `game_setup_cli` script or if installed, the `game-setup` cli. By default,
the script expects slurk to run on `localhost:5000`. If slurk runs on another machine, then you must provide
the `--slurk_host` and `--slurk_port` options. If you do not use the default admin token, then you must use
the `--token` option to provide the admin token for the game setup.

The script will create the game room, task and layout using the default admin token via slurks REST API. This will also
create three tokens: one for the game master, player and avatar. See the console output for these tokens. You can also
manually provide a name for the room.

### Prepare clients and bots

#### 1. Start the game master bot

Run the `game_master_cli --token <master-token>` script or the `game-master --token <master-token>` cli. This will
connect the game master with slurk. By default, this will create image-urls that point to `localhost:8000`.

#### 2. Start a browser for the 'Player'

Run a (private-mode) browser and go to `localhost:5000` and login as `Player` using the `<player-token>`.

#### 3. Start a browser for the 'Avatar' or start the avatar bot

**If you want to try the avatar perspective on your own**, then run a different (private-mode) browser and go
to `localhost:5000` and login as `Avatar` using the `<avatar-token>`.

**If the avatar is supposed to be your bot**, then run the `game_avatar_cli --token <avatar-token>` script or the
the `game-avatar --token <avatar-token>` cli. This will start the bot just like with the game master.

Note: This works best, when the game master is joining the room, before the avatar or the player. The order of player
and avatar should not matter. If a game seems not starting or the avatar seems not responding try to restart a game
session with the `/start` command in the chat window.

Another note: The dialog history will be persistent for the room. If you want to "remove" the dialog history, then you
have to create another room using the `game-setup` cli (or restart slurk and redo everything above). The simple avatar
does not keep track of the dialog history.

## B. Run everything with ngrok (temporarily externally available)

This setup will allow you to play temporarily with others over the internet.

First, do everything as in *Run everything on localhost: Prepare servers and data*.

### Prepare ngrok

1. Externalize the slurk server using `ngrok http 5000` which will give you something like `<slurk-hash>.ngrok.io`
1. Externalize the image server using `ngrok http 8000` which will give you something like `<image-hash>.ngrok.io`

### Prepare clients and bots

#### 1. Start the game master bot

Run the `game_master_cli --token <master-token> --image_server_host <image-hash>.ngrok.io --image_server_port 80`. This
will connect the game master with slurk. This will create image-urls that point to `<image-hash>.ngrok.io:80` which
makes the images accessible over the internet. If you run the game master on the same host as slurk, then the game
master will automatically connect to `localhost:5000`. If you run the game master on another machine than slurk, then
you probably want to also provide the `--slurk_host <slurk-hash>.ngrok.io` and `--slurk_port 80` options.

#### 2. Start a browser for the 'Player'

Run a (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Player` using the player token. You might
have to wait a minute until you can also connect as the second player.

#### 3. Start a browser for the 'Avatar'

Run a different (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Avatar` using the avatar token.
If the avatar is a bot, then the bot will have to use the token, when the connection is about to be established, just
like with the game master.
