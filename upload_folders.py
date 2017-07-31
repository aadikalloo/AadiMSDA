import os

folder_list = [x[0] for x in os.walk('/home/aadi_kalloo/cdsa_imgs/tiles/brca')]

for folder in folder_list[1:]:
  file_for_upload = folder[:-1]+'.zip'
  os.system('zip -qr '+file_for_upload+' '+folder)
  os.system("/bin/bash /home/aadi_kalloo/google-drive/google-drive-upload-master/upload.sh "+ file_for_upload +" brca")
  os.system("rm -r "+folder)
  os.system("rm "+file_for_upload)
