---
layout: default
---
# Introduction #
write introduction text

This tutorial explains how to use the efn git repo to learn built-in or user-specified exponential family models.  You should follow the [standard install](#standard-install) instructions if you only intend to learn EFNs for the built-in exponential families.  If you intend to write tensorflow code for your own exponential family class, you should use the [dev install](#dev-install).  

# Installation #
## Standard install<a name="standard-install"></a> ##
These installation instructions are for users interested in learning exponential families, which are already implemented in the [exponential families library](families.md).  Clone the git repo, and then go to the base directory and run the installer.
```bash
git clone https://github.com/cunningham-lab/efn.git
cd efn/
python setup.py install
```

## Dev install<a name="dev-install"></a> ##
If you intend to write some tensorflow code for your own exponential family, then you want to use the development installation.  Clone the git repo, and then go to the base directory and run the development installer.
```bash
git clone https://github.com/cunningham-lab/efn.git
cd efn/
python setup.py develop
```

