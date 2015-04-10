from PIL import Image
import pytesser.pytesser


if __name__=='__main__':
	image_file = 'myimage.tif'
	im = Image.open(image_file)
	text = image_to_string(im)
	text = image_file_to_string(image_file)
	text = image_file_to_string(image_file, graceful_errors=True)
	print "=====output=======\n"
	print text