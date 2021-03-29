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

We'll need to create a "virtual environment" where our LFP-Process
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

source /usr/local/bin/virtualenvwrapper.sh

```

Reload your terminal to apply. The last command “mkvirtualenv” should create a virtual environment which causes your prompt to read something like:

```bash
(lfp-process)user@host:folder$
```


```bash
python3 —version
```

## Issues with Python 3

By default we will install the latest version of python3. Historically the group uses Python v3.4. It is probably OK to use the latest, but, if we want to roll-back, tell Homebrew to install Python 3.4.x:

```bash
$ cd /usr/local
$ git checkout fedb343 /usr/local/Library/Formula/python3.rb
$ brew install python3
$ git reset --hard
```

**Warning:** A naive `brew install python3` without a `git checkout` first will
install the latest Python 3.x version (currently 3.4), which may *not* be
compatible with our codebase.

Verify that it installed properly:

```bash
$ python3 --version
Python 3.4.?
```


## Uninstall Python 3 and PIP

You need to deactivate the environment before removing python. This step works only if you installed python3 with brew to start. If the uninstall did not remove the link follow to brew to force unlink:

```bash
deactivate
rmvirtualenv lfp-process
brew uninstall --ignore-dependencies python3 --force
brew install python3
```

if you need to update Xcode to to run the brew also 
```bash
xcode-select --install
```

You need to repeat all the installation

if link failed:
brew link --overwrite python3


## Additional issues: Qt5 related-error

The following issue was identified when running the following command:

```bash

./tools/readFlagsProcessedData.py --start 2020-02-16-06:00:00 --end 2020-02-16-19:04:00 --userID <SOME USER ID HERE>

```

The solution is to go to "~/.matplotlib" then edit or created a file named matplotlibrc and save the following line "backend: Qt5Agg"
Namely:

```bash
cd ~/.matplotlib
vim matplotlibrc

# Insert the following line
backend: Qt5Agg

```

