from cProfile import label
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


st.sidebar.image("./app_logo.png")
st.sidebar.header("DurAIn-ni Webapp." + "\n" + "ระบบวิเคราะห์อาการป่วยของต้นทุเรียนด้วย AI")
def load_image(image_file):
    img = Image.open(image_file)
    return img


activities = ["Detection (วิเคราะห์โรค)", "About (เกี่ยวกับ)"]
choice = st.sidebar.selectbox("Select option.. (เลือกโหมด)",activities)

#set default size as image_scale x image_scale
def img_resize(input_path,img_size): # padding
  desired_size = img_size
  im = Image.open(input_path)
  im = ImageOps.exif_transpose(im) # fix image rotating
  width, height = im.size # get img_input size
  if (width == image_scale) and (height == image_scale):
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

image_scale = 768 # change image input / ouput size here

def get_detection(img_path):
 
  #Get_Idcard_detail(file_path=img_path)
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img) # fix image rotating
  width, height = img.size # get img_input size
  if (width == image_scale) and (height == image_scale):
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.6)
  else:
    #im = im.convert('L') #Convert to gray
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(image_scale)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (image_scale, image_scale))
    new_im.paste(img, ((image_scale-new_size[0])//2,
                        (image_scale-new_size[1])//2))
    pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.6)


    
    #st.write(new_im.size)

  

  try:
    labels, acc = pred_dict['detection']['labels'][0], pred_dict['detection']['scores'][0]
    acc = acc * 100
    if labels == "powder":
      labels = "โรคราแป้ง"
      treatment = "ฉีดพ่นสารอินทรีย์ ไอเอส ในอัตราส่วน 50ซีซี ต่อน้ำ 20ลิตร ทุก 3-7 วัน บำรุงต้นทุเรียน ให้เติบโต สมบูรณ์แข็งแรงอยู่เสมอ ด้วยสารอาหารพื้นฐาน ควรตรวจตราผลทุเรียนในแปลงปลูกอย่างสม่ำเสมอ ในช่วงที่พบว่ามีการเกิดโรคในสวนทุเรียนของตน หากพบเจอใบที่ร่วงแล้วมีราแป้งให้นำไปทำลายทิ้งทันที หากไม่ทำลายอาจเป็นสาเหตุให้เกิดการระบาดต่อไปได้"
    elif labels == "mg_loss":
      labels = "อาการขาดธาตุ แมกนีเซียม"
      treatment = "ผสมปุ๋ยทางดิน : ผสมผงแมกนีเซียมซัลเฟต กับ โดโลไมท์  ให้เข้ากันตามความเหมาะสม" + "\n" + "ผสมปุ๋ยทางใบ : ผงแมกนีเซียมผสมในน้ำแล้วฉีดรดน้ำต้นไม้ต้นนั้นไปได้เลย"
    elif labels == "n_loss":
      labels = "อาการขาดธาตุ ไนโตรเจน"
      treatment = "ผสมปุ๋ยทางดิน : ผสมปุ๋ย NPK ที่มีอัตราส่วนของค่า N มากที่สุด และสังเกตปริมาณการใช้ตามอาการของใบ" + "/n" + "ผสมปุ๋ยทางใบ : ใช้ปุ๋ยเคมีที่มีค่า N สูงๆ หรือ ใช้ยูเรียน้ำ สูตรไนโตรเจนสูง ผสมแล้วพ้นปุ๋ยนํ้า"
    elif labels == "blight":
      labels = "โรคใบไหม้"
      treatment = "ฉีดพ่นสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพ ยารักษาทางการค้า เช่น ปุ๋ยน้ำไตรโครเดอร์มา , จุลินทรีย์ปราบโรค (20 cc ต่อน้ำ 20ลิตร ทางใบ) หรือ สารกลุ่มรหัส 3 และสารกลุ่มรหัส 11 ผสมหรือสลับ ด้วยสารประเภทสัมผัส เช่น สารกลุ่มคอปเปอร์ แมนโคเซ็บ"
    elif labels == "spot":
      labels = "โรคใบจุด"
      treatment = "ฉีดพ่นสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพ ยารักษาทางการค้า เช่น ปุ๋ยน้ำไตรโครเดอร์มา , จุลินทรีย์ปราบโรค (20 cc ต่อน้ำ 20ลิตร ทางใบ) หรือ สารกลุ่มรหัส 3 และสารกลุ่มรหัส 11 ผสมหรือสลับ ด้วยสารประเภทสัมผัส เช่น สารกลุ่มคอปเปอร์ แมนโคเซ็บ"

    st.success(f"มีโอกาสเป็น : {labels}  {round(acc, 2)} % .")
    st.write('วิธีการรักษา {} : '.format(labels))
    st.write(treatment)
  except IndexError:
    st.error("ไม่พบโรคในใบทุเรียน กรุณาตรวจเช็คภาพ แล้ว ใส่รูปใหม่อีกรอบ")

    labels = "None"
    acc = 0

def get_img_detection(img_path):
   
  #Get_Idcard_detail(file_path=img_path)
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img) # fix image rotating
  width, height = img.size # get img_input size
  if (width == image_scale) and (height == image_scale):
    new_im = img
  else:
    #im = im.convert('L') #Convert to gray
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(image_scale)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (image_scale, image_scale))
    new_im.paste(img, ((image_scale-new_size[0])//2,
                        (image_scale-new_size[1])//2))
  
  pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.6)


  return pred_dict['img']






if choice =='About (เกี่ยวกับ)' :
    st.header("About... (เกี่ยวกับ)")

    st.subheader("DurAIn-ni คืออะไร ?")
    st.write("- เป็นระบบที่วิเคราะห์โรคที่เกิดได้จากต้นทุเรียนผ่านใบ พร้อม แนะนำวิธีในการรักษาเบื้องต้น โดยใช้ ระบบ AI")

    st.subheader("DurAIn-ni ทำอะไรได้บ้าง ?")
    st.write("- วิเคราะห์โรคจากต้นทึเรียนโดยใช้ใบ (Object detection)")
    st.write("- ให้คำแนะนำในการรักษาเบื้องต้น")
    st.write("- โดยในปัจจุบันสามารถวิเคราะหได้ 5 โรค หลักๆ ได้แก่ " + "\n" + "  1. โรคราแป้ง (Powdery mildew disease)" + "\n" + "  2. โรคใบไหม้ (Leaf blight disease)" + "\n" + "  3. โรคใบจุด (Leaf spot disease)" + "\n" + "  4. อาการขาดธาตุ แมกนีเซียม (Magnesium deficiency)" + "\n" + "  5. อาการขาดธาตุ ไนโตรเจน (Nitrogen deficiency)")

    st.subheader("ทำไมถึงต้องเลือกใช้ ADurAIn-ni ")
    st.write("- สามารถใช้วินิจฉัยโรคที่เกิดจากต้นทุเรียนเบื้องต้นได้ด้วยตัวเองในทันที โดยไม่จำเป็นที่จะต้องมีผู้เชี่ยวชาญในการวิเคราะห์ตลอดเวลา พร้อมบอกรายละเอียดโรค และ วิธีการรักษาในเบื้องต้น ดังนั้น  เกษตรกร และ คนทั่วไป สามารถวิเคราห์โรคที่เกิดจากต้นทุเรียน และ สามารถที่จะรักษาได้ได้โดยทันท่วงที ")



    st.subheader("คำแนะนำในการใช้งาน")
    st.write("- ภาพไม่ควรมีแสงที่สว่างมากเกืนไป และ มืดเกินไป มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อนเอาได้")#4
    st.write("- ภาพไม่ควรที่จะอยู่ไกลเกินไป และ ควรมีความชัด มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อน หรือ ไม่สามารถตรวจจับได้")#5

    st.subheader("รายละเอียดเพิ่มเติม")
    st.write('[โรคพื้นฐานที่เกิดในต้นทุเรียน](https://kasetgo.com/t/topic/401483)')
    st.write('[สถิติการส่งออกทุเรียนของไทย](http://impexp.oae.go.th/service/export.php?S_YEAR=2560&E_YEAR=2565&PRODUCT_GROUP=5252&PRODUCT_ID=4977&wf_search=&WF_SEARCH=Y)')

    
     
       
elif choice == "Detection (วิเคราะห์โรค)":
    st.header("Detection (วิเคราะห์โรค)")

    image = st.file_uploader(label = "Upload Durian leaf here.. (ใส่รูปภาพตรงนี้)",type=['png','jpg','jpeg'])
    if image is not None:
        st.write("## Detection result.. (สรุปผลการวิเคราะห์)")
        new_img = img_resize(image, image_scale)
        st.image(get_img_detection(image))
        t1 = time.perf_counter()
        get_detection(image)
        t2 = time.perf_counter()
        st.write('time taken to run: {:.2f} sec'.format(t2-t1))







    else:
        st.write("## Waiting for image.. (รอผู้ใช้งานใส่รูปภาพ)")
        st.image('ania.png')

    st.caption("Made by Tanaanan .M")




st.sidebar.markdown('---')

st.sidebar.subheader('More image for test..')
st.sidebar.write('[Github img test set.](https://github.com/Tanaanan/DurAIn-ni-/tree/main/testset_imgs)')

st.sidebar.subheader('Made by Tanaanan .M')
st.sidebar.write("Contact : mjsalyjoh@gmail.com")
