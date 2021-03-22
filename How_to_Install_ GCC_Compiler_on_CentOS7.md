```
sudo yum --disablerepo=nvidia-docker install centos-release-scl
sudo yum install devtoolset-7
## insert the next to your .bashrc
scl enable devtoolset-7 bash
## check!
gcc -v
gcc version 7.3.1 20180303 (Red Hat 7.3.1-5) (GCC)
```
