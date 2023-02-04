# Mô phỏng THz trên ns3

## Cài đặt mô phỏng:

```
git clone https://github.com/rgrunbla/ns-3-33.git
git clone https://github.com/UN-Lab/thz.git
cp -r thz/ ns-3-33/contrib/
cd ns-3-33
./waf configure --disable-werror
make
```

Copy file nano-thz.cc vào `scratch`

## Chạy mô phỏng:

```
./waf --run scratch/nano-thz
```
