#!/usr/bin/python
import argparse
import sys
from lib.PPM import PPM
from lib.CVStream import CVStream

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Manually and extremely inefficiently process some images (yay!)')
  parser.add_argument('infile', nargs=1, metavar='file', 
    type=str, help='input ppm')
  parser.add_argument('-o', '--out', 
    metavar='file', type=str, help='output ppm')
  parser.add_argument('-a', '--action', metavar='operation', type=str,
    choices=[
      'GSRed', 'GSGreen', 'GSBlue', 'GSAverage', 'GSWeightedAverage',
      'scale', 'bilinearScale', 'rotate', 'stream'
    ],
    required=True,
    help='''available operations: GSRed, GSGreen, GSBlue, GSAverage,
    GSWeightedAverage, scale, bilinearScale, rotate
    '''
  )
  parser.add_argument('-r', '--angle', metavar='angle', type=int,
    help='angle of rotation', required=False)
  parser.add_argument('-s', '--size', metavar='size', type=str,
    help='size to scale image to. e.g. \'1234x5678\'', required = False)
  args = parser.parse_args()
  
  action = args.action
  file = None
  if action != 'stream':
    file = args.infile[0]
  else:
    cv = CVStream()
    cv.stream_video()
  
  if action == 'rotate':
    if args.angle is None:
      print 'Rotate action requires an angle.'
      sys.exit(-1)

    angle = args.angle
    ppm = PPM(file).rotateImage(angle).write(args.out)

  if action == 'GSRed':
    PPM(file).grayscaleRed().write(args.out)

  if action == 'GSGreen':
    PPM(file).grayscaleGreen().write(args.out)

  if action == 'GSBlue':
    PPM(file).grayscaleBlue().write(args.out)

  if action == 'GSAverage':
    PPM(file).grayscaleAverage().write(args.out)

  if action == 'GSWeightedAverage':
    PPM(file).grayscaleWeightedAverage().write(args.out)

  if action == 'scale' or action == 'bilinearScale':
    if args.size is None:
      print 'Scale action requires a size.'
      sys.exit(-1)

    size = args.size.split('x')
    x, y = int(size[0]), int(size[1])
    if action == 'scale':
      PPM(file).scale(x, y).write(args.out)
    else:
      PPM(file).scaleBilinear(x, y).write(args.out)

  # print args

# ppm = PPM('./d20WAGS.ppm')
# ppm.scaleUp(2, 'output.ppm').write()
