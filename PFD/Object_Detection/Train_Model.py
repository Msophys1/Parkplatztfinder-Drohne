# Befehl in Console ausführen
# Im PFD Directory sein

# Mit ssd_mobilenet_v2_320x320_coco17_tpu-8
# python models/research/object_detection/model_main_tf2.py --pipeline_config_path=Object_Detection/ssd_mobilenet_v2_320x320_coco17_tpu-8.config  --model_dir=Object_Detection/training --alsologtostderr

# Mit ssd_efficientdet_d0_512x512_coco17_tpu-8
# python models/research/object_detection/model_main_tf2.py --pipeline_config_path=Object_Detection/ssd_efficientdet_d0_512x512_coco17_tpu-8.config  --model_dir=Object_Detection/training --alsologtostderr


# Der Code von seiner Webseite ist falsch!
# Website: https://github.com/BenGreenfield825/Tensorflow-Object-Detection-with-Tensorflow-2.0


# Model exportieren
# python models/research/object_detection/exporter_main_v2.py --trained_checkpoint_dir=Object_detection/training  --pipeline_config_path=Object_Detection/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --output_directory Object_Detection/inference_graph