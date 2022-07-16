import streamlit as st  #Web App
from PIL import Image, ImageOps #Image Processing
import time
from unittest import result
from pythainlp.util import isthai
import numpy as np
from icevision import tfms
from icevision.models import model_from_checkpoint
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


st.sidebar.image("./logo.png")
st.sidebar.header("Du-rAIn san Web app")
def load_image(image_file):
    img = Image.open(image_file)
    return img


activities = ["Detection", "About"]
choice = st.sidebar.selectbox("Select option..",activities)

#set default size as 1280 x 1280
def img_resize(input_path,img_size): # padding
  desired_size = img_size
  im = Image.open(input_path)
  im = ImageOps.exif_transpose(im) # fix image rotating
  width, height = im.size # get img_input size
  if (width == 1280) and (height == 1280):
    new_im = im
  else:
    #im = im.convert('L') #Convert to gray
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

  return new_im

checkpoint_path = "./Leaf_disease_rcnn.pth"

checkpoint_and_model = model_from_checkpoint(checkpoint_path, 
    model_name='mmdet.faster_rcnn', 
    backbone_name='resnet101_fpn_2x',
    img_size=512, 
    is_coco=False)

model_type = checkpoint_and_model["model_type"]
backbone = checkpoint_and_model["backbone"]
class_map = checkpoint_and_model["class_map"]
img_size = checkpoint_and_model["img_size"]
#model_type, backbone, class_map, img_size

model = checkpoint_and_model["model"]

device=next(model.parameters()).device

img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])

def get_detection(img_path):
 
  #Get_Idcard_detail(file_path=img_path)
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img) # fix image rotating
  width, height = img.size # get img_input size
  if (width == 1280) and (height == 1280):
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.6)
  else:
    #im = im.convert('L') #Convert to gray
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(1280)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (1280, 1280))
    new_im.paste(img, ((1280-new_size[0])//2,
                        (1280-new_size[1])//2))
    pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.6)


    
    #st.write(new_im.size)

  

  try:
    labels, acc = pred_dict['detection']['labels'][0], pred_dict['detection']['scores'][0]
    acc = acc * 100
    st.success(f"Result : {labels} with {round(acc, 2)}% confidence.")
  except IndexError:
    st.error("No disease found! ; try to recheck image again..")
    labels = "None"
    acc = 0

def get_img_detection(img_path):
   
  #Get_Idcard_detail(file_path=img_path)
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img) # fix image rotating
  width, height = img.size # get img_input size
  if (width == 1280) and (height == 1280):
    new_im = img
  else:
    #im = im.convert('L') #Convert to gray
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(1280)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (1280, 1280))
    new_im.paste(img, ((1280-new_size[0])//2,
                        (1280-new_size[1])//2))
  
  pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.6)


  return pred_dict['img']






if choice =='About' :
    st.header("About...")

    st.subheader("AOC คืออะไร ?")
    st.write("- เป็นระบบที่สามารถคัดกรองผลตรวจเชื้อของ COVID-19 ได้ผ่าน ที่ตรวจ ATK (Antigen Test Kit) ควบคู่กับบัตรประชาชน จากรูปภาพได้โดยอัตโนมัติ")

    st.subheader("AOC ทำอะไรได้บ้าง ?")
    st.write("- ตรวจจับผลตรวจ ATK (Obj detection)")
    st.write("- ตรวจจับชื่อ-นามสกุล (OCR)")
    st.write("- ตรวจจับเลขบัตรประชาชน (OCR)")

    st.subheader("AOC ดีกว่ายังไง ?")
    st.write("จากผลที่ได้จากการเปรียบเทียบกันระหว่าง model (AOC) กับ คน (Baseline) จำนวน 30 ภาพ / คน ได้ผลดังนี้")
    st.image("./acc_table.png")
    st.write("จากผลที่ได้สรุปได้ว่า ส่วนที่ผ่าน Baseline และมีประสิทธิภาพดีกว่าคัดกรองด้วยคนคือ ผลตรวจ ATK ได้ผลที่ 100 %, เลขบัตรประชน ได้ผลที่ 100 % และ ความเร็วในการคัดกรอง ได้ผลที่ 4.84 วินาที ซึ่งมีความเร็วมากกว่า 81% เมื่อเทียบกับคัดกรองด้วยคน ถือว่ามีประสิทธิภาพที่สูงมากในการคัดกรอง และ มีประสิทธิภาพมากกว่าการคัดแยกด้วยมนุษย์")
    st.write("** ความเร็วที่โมเดลทำได้อาจไม่ตรงตามที่ deploy บนเว็บ เนื่องจากในเว็บ ไม่มี GPU ในการประมวลผลอาจทำให้โมเดลใช้เวลาในการประมวลที่นานกว่าตอนใช้ GPU")


    st.subheader("คำแนะนำในการใช้งาน")
    st.write("- ในการใช้งานให้ถ่ายรูปภาพบัตรประชาชนในแนวตั้งเท่านั้น เนื่องจากถ้าเป็นแนวอื่นอาจทำให้การตรวจจับคลาดเคลื่อนเอาได้")#3
    st.write("- ภาพไม่ควรมีแสงที่สว่างมากเกืนไป และ มืดเกินไป มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อนเอาได้")#4
    st.write("- ภาพไม่ควรที่จะอยู่ไกลเกินไป และ ควรมีความชัด มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อน หรือ ไม่สามารถตรวจจับได้")#5

    st.subheader("รายละเอียดเพิ่มเติม")
    st.write('[Medium blog](https://medium.com/@mjsalyjoh/atk-ocr-classification-aoc-%E0%B8%A3%E0%B8%B0%E0%B8%9A%E0%B8%9A%E0%B8%84%E0%B8%B1%E0%B8%94%E0%B8%81%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%9C%E0%B8%A5%E0%B8%95%E0%B8%A3%E0%B8%A7%E0%B8%88-atk-%E0%B9%81%E0%B8%A5%E0%B8%B0-%E0%B8%9A%E0%B8%B1%E0%B8%95%E0%B8%A3%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%8A%E0%B8%B2%E0%B8%8A%E0%B8%99-fa32a8d47599)')
    st.write('[Github Link](https://github.com/Tanaanan/AOC_ATK_OCR_Classification)')

    
     
       
elif choice == "Detection":
    st.header("Durain disease detection (leaf)")

    image = st.file_uploader(label = "Upload Durian leaf here..",type=['png','jpg','jpeg'])
    if image is not None:
        new_img = img_resize(image, 1280)
        st.image(get_img_detection(image))
        t1 = time.perf_counter()
        get_detection(image)
        t2 = time.perf_counter()
        st.write('time taken to run: {:.2f} sec'.format(t2-t1))







    else:
        st.write("## Waiting for image..")
        st.image('spy-x-family-anya-heh-anime.jpeg')

    st.caption("Made by Tanaanan .M")




st.sidebar.markdown('---')
st.sidebar.subheader('Made by Tanaanan .M')
st.sidebar.write("Contact : mjsalyjoh@gmail.com")