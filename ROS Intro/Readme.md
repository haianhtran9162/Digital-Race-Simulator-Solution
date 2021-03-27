# Hướng dẫn làm việc cơ bản với ROS
## Với simulator
simulator cung cấp một số topic để giao tiếp truyền, nhận dữ liệu dưới đây

/tênđội/set_speed: Topic được publish từ ROS_node được định nghĩa dưới dạng số thực (Float32). Là tốc độ xe cần đạt. 

/tênđội/set_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền góc lái của xe. 

/tênđội/set_camera_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền quay của camera.

/tênđội/camera/rgb/compressed: Topic dùng để subcribe ảnh rgb thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.

/tênđội/camera/depth/compressed: Topic dùng để subcribe ảnh depth thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.

tên đội đặt giống tên đội khi điền vào simulator

## Để kiểm tra các topic
B1:
`$ roslaunch goodgame_fptu_dl video_stream.launch `

B2: mở simulator

B3: xem các topic đang được publisher - subscriber

`$ rostopic list ` 

B4: Nếu thấy xuất hiện 5 topic phía trên thì simulator đã kết nối thành công với ROS
    
Kiểm tra các topic subscriber
    
`$ rostopic echo [tên topic]`
    
 Ví dụ: 
`$ rostopic echo /g2_never_die/camera/rgb/compressed`
    
 Kiểm tra các topic publisher
    
`$ rostopic pub [tên topic] [kiểu dữ liệu] [giá trị dữ liệu]`
    
 Ví dụ:
`$ rostopic pub /g2_never_die/set_speed std_msgs/Float32 40`
    


