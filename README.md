## Digital-Race-Simulator-Solution
![maxresdefault](https://user-images.githubusercontent.com/48562455/112740137-a58fb800-8fa4-11eb-8da9-a3c4d58b7546.jpg)
## Thông tin hệ thống chúng tôi sử dụng
Ubuntu 18.4 LTS\
Python 2.7\
CUDA 10.0\
CUDnn 4.6.0\
ROS Melodic\
Tensortflow-gpu 1.15\
Keras 2.2.4
## Hướng dẫn cài đặt ROS Melodic
1. Cài đặt ROS Melodic:
  http://wiki.ros.org/melodic/Installation/Ubuntu \
  Chú ý: cần cài đặt bản đầy đủ. Ví dụ với ROS Lunar thì cài đặt bản đầy đủ như sau:\
  `$sudo apt-get install ros-lunar-desktop-full`\
  Hoàn Thiện đầy đủ các bước từ mục 1.1 đến 1.7
2. Tạo ROS Workspace
  `$ mkdir -p ~/catkin_ws/src`\
  `$ cd ~/catkin_ws/`\
  `$ catkin_make`\
  `$ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc`\
  `$ source ~/.bashrc`
3. Cài đặt rosbridge-suite\
`$ sudo apt-get install ros-lunar-rosbridge-server`
## Download Simulator
Download link: http://bit.ly/cds-simulator-2019

Simulator xây dựng  sa hình thật: https://github.com/vietanhdev/cuoc-dua-so-fpt-round2-sim
## Hướng dẫn chạy source code
1. Clone/Download code
2. Sao chép thư mục goodgame_fptu_dl vào thư mục catkin_ws vừa tạo trên
3. Thức hiện chạy roslaunch `$ roslaunch goodgame_fptu_dl goodgame.launch`
4. (Optional) Nếu hệ thống yêu cầu cấp quyền cho một số file chạy thì cd vào thư mục chứa file đó và cấp quyền: `$ sudo chmod +x <tên_file>`
5. Mở simmulator và điền với Team name : **g2_never_die** và Port : **ws://127.0.0.1:9090**
6. Ấn Start
## Issues
### Lỗi **no module named rospy_message_converter** 

Run command `sudo apt install ros-melodic-rospy-message-converter`

### Lỗi **no module named websocket** 

Run command `pip2 install websocket_client --user`

### Lỗi disconnect websocket 

Run command 
`pip uninstall tornado`
`sudo apt-get install ros-melodic-rosbridge-suite`

# Contact / Info
If you are interested in the detailed development process of this project, you can contact me at email address: stephen.t.vu@hotmail.com or datvthe140592@fpt.edu.vn

**Contributors:**

**Dat Vu (Leader)** | [Email](mailto:stephen.t.vu@hotmail.com) | [Github](https://www.github.com/datvuthanh) | [Website](https://datvuthanh.github.io/)

<img src="./images/datvu.jpg" alt="Drawing" width="80" height="80"/>

**Hai Anh Tran** | [Email](mailto:anhthhe141545@fpt.edu.vn) | [Github](https://github.com/AnhTH-FUHN)

<img src="./images/haianh.jpg" alt="Drawing" width="80" height="80"/>

**Tra Dinh** | [Email](mailto:trandhe140661@fpt.edu.vn) 

<img src="./images/tradinh.png" alt="Drawing" width="80" height="80"/>


**Huy Phan** | [Email](mailto:HuyPQHE141762@fpt.edu.vn) 

<img src="./images/huyphan.png" alt="Drawing" width="80" height="80"/>

