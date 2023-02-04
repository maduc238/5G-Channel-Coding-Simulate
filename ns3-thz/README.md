# Mô phỏng THz trên ns3

Dependence:
```shell
sudo apt-get install gcc g++ python python-dev mercurial bzr gdb valgrind gsl-bin libgsl0-dev flex bison tcpdump sqlite sqlite3 libsqlite3-dev libxml2 libxml2-dev libgtk2.0-0 libgtk2.0-dev uncrustify doxygen graphviz imagemagick texlive texlive-latex-extra texinfo dia texlive texlive-latex-extra texlive-extra-utils texi2html
```

## Cài đặt mô phỏng:

```shell
git clone https://github.com/rgrunbla/ns-3-33.git
git clone https://github.com/UN-Lab/thz.git
cp -r thz/ ns-3-33/contrib/
cd ns-3-33
./waf configure --disable-werror
make
```

Copy file nano-thz.cc vào `scratch`

## Chạy mô phỏng:

```shell
./waf --run scratch/nano-thz
```
