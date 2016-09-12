import numpy as np
import cv2
import sys

def nextBit(in_str):
	for ch in in_str:
		c = ord(ch)
		yield c & 0x80 > 0
		yield c & 0x40 > 0
		yield c & 0x20 > 0
		yield c & 0x10 > 0
		yield c & 0x08 > 0
		yield c & 0x04 > 0
		yield c & 0x02 > 0
		yield c & 0x01 > 0

def decode(result):
	length = len(result)
	ret = ''
	if length % 8 != 0:
		result = result[:length - (length % 8)]
	for i in range(0, length, 8):
		num = 0
		for idx, n in enumerate([r for r in result[i:i+8]]):
			num += n << (7 - (idx % 8))
		if num != 0:
			ret += chr(num)
	return True, ret

image = None
output = None
text = None
action = None
try:
	action = sys.argv[1]
	image = sys.argv[2]
	if action != 'encode' and action != 'decode':
		raise Exception()
	if action == 'encode':
		text = sys.argv[3]
		output = sys.argv[4]

		if text == '-':
			text = sys.stdin.read()
			# print text
except:
	print 'USAGE: hide (encode input_image (text | -) output_image_name | decode input_image)'
	sys.exit(1)

image = cv2.imread(image)
stop_pixel = np.full((3,), 0, dtype=np.uint8)

width = image.shape[1]
height = image.shape[0]
result = []
i = 0

if action == 'encode':
	bits = iter(list(nextBit(text)))

while i < width * height:
	if action == 'encode':
		y, x, z = i % height, i // width, i % 3
		val = image[y, x, z]
		try:
			bit = bits.next()
			image[y, x, z] = np.uint8((val & 0xFE) | bit)
		except:
			i += 1
			y, x = i % height, i // width
			image[y, x] = stop_pixel
			break
	elif action == 'decode':
		y, x, z = i % height, i // width, i % 3
		px = image[y, x]
		if px[0] == 0 and px[1] == 0 and px[2] == 0:
		# if np.all([image[y, x], stop_pixel]):
			break

		result.append(image[y, x, z] & 0x01 > 0)

	i += 1 

# print image
# cv2.imshow('image', image)
# cv2.waitKey(0)

if action == 'encode':
	cv2.imwrite(output, image)
elif action == 'decode':
	ret, result = decode(result)
	print '--- DECODED TEXT ---\n', result, '\n--------------------'

print 'DONE'
