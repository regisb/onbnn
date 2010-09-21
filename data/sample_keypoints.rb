#!/usr/bin/ruby
# Usage: ruby ./sample_keypoints <image directory> <descriptor>"
# Samples interest points from all jpg images in a given directory 
# and saves them under the form of a machine-readable text file
# Descriptor argument can be one of the following types: 
# (see http://staff.science.uva.nl/~ksande/research/colordescriptors/readme)
#     RGB histogram (rgbhistogram)
#     ROpponent histogram (opponenthistogram)
#     Hue histogram (huehistogram)
#     rg histogram (nrghistogram)
#     Transformed Color histogram (transformedcolorhistogram)
#     Color moments (colormoments)
#     Color moment invariants (colormomentinvariants)
#     SIFT (sift)
#     HueSIFT (huesift)
#     HSV-SIFT (hsvsift)
#     OpponentSIFT (opponentsift)
#     rgSIFT (rgsift)
#     C-SIFT (csift)
#     RGB-SIFT(rgbsift), equal to transformed color SIFT (transformedcolorsift)

##################### Useful parameters ####################
extension = ".jpg"          # Change this if you want to sample points from images of another kind
detector = "harrislaplace"  # You can set this to "densesampling" instead
executable = "./colorDescriptor" # Path to the color descriptor sampling executable
###########################################################

if ARGV.size() < 2 || ARGV[0] == "-h"
  puts "Usage: ruby ./sample_keypoints <image directory> <descriptor>"
  exit
end

img_dir = File.expand_path(ARGV[0])
descriptor = ARGV[1]

search = File.join(img_dir, "*#{extension}")
Dir.glob(search).sort().each{|img|
  # Output file
  out_file = File.join(img_dir, File.basename(img, extension) + "___" + descriptor + ".txt")
  puts "Producing #{out_file}..."
  # Build keypoint-sampling command
  cmd = "#{executable} #{img} --detector #{detector} --descriptor #{descriptor} --output #{out_file}"
  # Execute command
  `#{cmd}`
}
