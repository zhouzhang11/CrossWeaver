
# from ultralytics import YOLO_m

# model = YOLO_m("cvpr.yaml")
# model.train(data="drone_vehicle_m.yaml",
#             epochs=50,
#             patience=30,
#             batch=8,
#             imgsz=800,
#             device=3 ,
#             r_init=9,
#             r_target=6,
#             adalora=True,
#             project="DroneVehicle",
#             name='yolov8s_new_model_newdataset',
#             pretrained= False,
#             optimizer='auto',
#             seed=0,
#             freeze=None,
#             close_mosaic = 5,
#             lrf = 0.005,
#             warmup_epochs = 5.0,
#             )
# from ultralytics import YOLO_m
# model = YOLO_m("Ablation/yolov8m/weights/last.pt")
# # model.load("./ultralytics/yolov8s.pt")  # load weights
# model.train(data="drone_vehicle_m.yaml",
#             epochs=150,
#             patience=60,
#             resume=True,
#             exist_ok=True,
#             batch=8,
#             imgsz=800,
#             device=3, 
#             r_init=9,
#             r_target=6,
#             adalora=True,
#             project="Ablation",
#             name='yolov8m',
#             pretrained= False,
#             optimizer='auto',
#             seed=0,
#             freeze=None,
            
#             cos_lr = True,
#             close_mosaic = 10,
#             lrf = 0.005,
#             warmup_epochs = 5.0,
#             lr0 = 0.01,
#             workers = 8,

#             )


from ultralytics import YOLO_m

model = YOLO_m("cvpr.yaml")
# model.load("./ultralytics/yolov8s.pt")  # load weights
model.train(data="drone_vehicle_m.yaml",
            epochs=150,
            # resume = True,
            patience=60,
            batch=8,
            imgsz=800,
            device=0, 
            r_init=9,
            r_target=6,
            adalora=True,
            project="Supply",
            name='supply',
            pretrained= False,
            optimizer='auto',
            seed=0,
            freeze=None,
            mosaic =0,
            cos_lr = True,
            close_mosaic = 10,
            lrf = 0.001,
            warmup_epochs = 10.0,
            lr0 = 0.01,
            workers = 8,

            )
