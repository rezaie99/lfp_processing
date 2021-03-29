# LFP Processing
This respoity is design for analysis of LFP and Video for rodent.
## Configure terminal and shell (bash, zsh)

Default shell in MacOS has been bash, MacOS Catalina update switched this to zsh
User preference for which to use, but some scripts have been setup assuming bash. To switch shells, open a terminal and enter:

```bash
$ chsh -s /bin/bash
```

Then close and open a new terminal to apply. To revert back to zsh, /bin/zsh.

The configuration file for bash is ~/.bash_profile. For zsh it is ~/.zshrc. You may need to create these if they don’t exist. These files store default settings for your shell which are loaded at startup.


## Optional - configure zsh with oh-my-zsh

The default MacOS terminal and zsh/bash shell are pretty bare-bones. If using zsh, a simple way to get many useful features such as auto-complete, Git support, etc, is to use “oh-my-zsh”: https://ohmyz.sh . Use the install commands there to grab it. Once installed, it will update your .zshrc file, which you can then add user options to. See below for info:

https://scriptingosx.com/2019/06/moving-to-zsh/
https://www.freecodecamp.org/news/how-to-configure-your-macos-terminal-with-zsh-like-a-pro-c0ab3f3c1156/ 

Note that oh-my-zsh includes built-in support for many features that would have to be added manually in bash etc. Make sure to include “git” in the plugins in the .zsrhc: plugins=(git) . You will need to re-load the terminal to apply changes.


## Install virtual environment

We'll need to create a "virtual environment" where our WHOOP-specific
packages can live. Install the virtualenv management tool first using pip3 to install. Note that the location of the .sh file may differ from below.  You will see the location of it as one of the output lines from the installation.

```bash
$ pip3 install virtualenvwrapper
```

We need to source this virtualenv in our config files so we load it by default. If using bash (or edit ~/.bash_profile):

```bash
$ echo '. /usr/local/bin/virtualenvwrapper.sh' >> ~/.bash_profile
```

If using zsh, add:

```bash
source /usr/local/bin/virtualenvwrapper.sh
```

At the bottom of your .zshrc file.

We also need to update some path variables before we create the virtual environment. Add the following to your .bash_profile or .zshrc file, at the bottom, but BEFORE sourcing virtualenvwrapper.sh above:

```bash
export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
export PATH=/Envs:$PATH
export PATH=/Envs/whoop-sigproc/bin:/usr/local/bin:/usr/local/sbin:$PATH

source /usr/local/bin/virtualenvwrapper.sh

mkvirtualenv whoop-sigproc --python $(which python3)
```

Reload your terminal to apply. The last command “mkvirtualenv” should create a virtual environment which causes your prompt to read something like:

```bash
(whoop-sigproc)user@host:folder$
```

If the terminal does not prompt (whoop-sigproc) the issue could be related to an incorrect default version of python. Try:

```bash
python3 —version
```
