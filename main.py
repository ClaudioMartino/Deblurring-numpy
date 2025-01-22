from functions import *
from plot import *
import argparse
from os import listdir
from os.path import splitext

# Arguments
valid_inputs = [f for f in listdir('images/sample_input/') if (splitext(f)[1] == '.ppm' or splitext(f)[1] == '.pgm')]
valid_inputs.append('all')

parser = argparse.ArgumentParser(description="Deblurring", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input-image", choices=valid_inputs, default=["lena.ppm"], nargs="+", help="Input image")
parser.add_argument("-k", "--kernel-size",    type=int, nargs="+", help="Kernel size to blur", default=[3])
parser.add_argument("-k2", "--kernel-size-2", type=int, nargs="+", help="Kernel size to de-blur")
parser.add_argument("-t", "--threshold",      type=float, default=0.07, help="Inversion threshold")
parser.add_argument("-s", "--plot-step",      type=float, default=0.01, help="Plot step")
parser.add_argument("-p", "--plot", action='store_true', help="Run the SNRs for different thresholds and plot the result")
parser.add_argument("--no-images",  action='store_true', help="Stop the creation of the image files (not the plots)")
parser.add_argument("--no-symm",    action='store_true', help="Perform the operations without symmetrization")
parser.add_argument("--precise",    action='store_true', help="Use the precise 2D Gaussian function as kernel")

args = parser.parse_args()
config = vars(args)

print(config)

# Directories
images_dir = "images/"
input_images_dir = images_dir + "sample_input/"
output_images_dir = images_dir + "results/"

# Read arguments
input_image_name_arr = config['input_image']
if(input_image_name_arr == ['all']):
  input_image_name_arr = valid_inputs[:-1]

kernel_sizes = config['kernel_size']

kernel_sizes_2 = config['kernel_size_2']
if(kernel_sizes_2 == None):
  kernel_sizes_2 = kernel_sizes

assert(len(kernel_sizes) == len(kernel_sizes_2))

threshold = config['threshold']

plot = config['plot']

plot_step = config['plot_step']

no_images = config['no_images']
no_symm = config['no_symm']
precise_kernel = config['precise']

valid_kernel_sizes = [3, 5, 7]
for k in kernel_sizes:
  if (precise_kernel == False and k not in valid_kernel_sizes) or k % 2 == 0:
    raise Exception("Invalid kernel sizes!") 

# Loop over each image
snrs_kernel_image = []
for input_image_name in input_image_name_arr:
  # Ouput file names
  if(not no_images):
    grayscale_image = create_file_name(output_images_dir, input_image_name, "gray")
    if(not no_symm):
      symmetric_image = create_file_name(output_images_dir, input_image_name, "symm")
    dft_image = create_file_name(output_images_dir, input_image_name, "dft" )
    idft_image = create_file_name(output_images_dir, input_image_name, "idft")

  # Open input file as array
  print("Opening " + input_image_name)
  input_image = input_images_dir + input_image_name
  # TODO open pgm
  if(splitext(input_image_name)[1] == '.ppm'):
    rgb_image_array = open_ppm_as_array(input_image)
    height, width = rgb_image_array.shape
    width //= 3
  else:
    grayscale_image_array = open_pgm_as_array(input_image)
    height, width = grayscale_image_array.shape
  print("Height: " + str(height) + "; Width: " + str(width))
  
  # Convert RGB array to grayscale
  if(splitext(input_image_name)[1] == '.ppm'):
    grayscale_image_array = rgb_to_grayscale(rgb_image_array)
    if(not no_images):
      save_pgm_file(grayscale_image_array, grayscale_image)
      print("> Created grayscale " + grayscale_image)

  # Symmetrization to avoid introduction of high freqs
  if(not no_symm):
    symm_image_array = symmetrization(grayscale_image_array)

  # Loop over each kernel size (blur, de-blur)
  snrs_kernel = []
  for kernel_size, kernel_size_2 in zip(kernel_sizes, kernel_sizes_2):
    # Create file names
    if(not no_images):
      blurred_image        = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "blur")
      blurred_image_f      = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "blur_f")
      dft_blur_image       = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_blur")
      dft_blur_image_f     = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_blur_f")
      dft_kernel_image     = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_kernel")
      #inverse_kernel_image = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "inv_kernel")
      deconv_image         = create_file_name(output_images_dir, str(kernel_size) + '_' + str(kernel_size_2) + '_' + input_image_name, "deconv")
      dft_deconv_image     = create_file_name(output_images_dir, str(kernel_size) + '_' + str(kernel_size_2) + '_' + input_image_name, "dft_deconv")
   
    # 2D FFT of input image
    if(not no_symm):
      dft_image_array = np.fft.fft2(symm_image_array)
    else:
      dft_image_array = np.fft.fft2(grayscale_image_array)

    if(not no_images):
      save_pgm_file(rearrange_FFT(dft_image_array), dft_image)
      print("> Created abs log scaled DFT " + dft_image)

    # Inverse 2D FFT of input image
    #idft_image_array = np.fft.ifft2(dft_image_array)
    #save_pgm_file(idft_image_array[0:width, 0:height].real.astype(np.uint8), idft_image)
    #print("> Created IDFT " + idft_image)

    # Blurring: convolution in space domain
    kernel_blur = gaussian_blur_kernel(kernel_size, precise_kernel)
    #kernel_blur = motion_blur_kernel(kernel_size)
    blurred_image_array = convolution(grayscale_image_array, kernel_blur, 1)
    if(not no_images):
      save_pgm_file(blurred_image_array, blurred_image)
      print("> Created blurred " + blurred_image)

      diff_image_array = np.abs((grayscale_image_array).astype(np.int16) - (blurred_image_array).astype(np.int16)).astype(np.uint8)
      heatmap(diff_image_array, 'heatmap_blur_' + str(kernel_size) + '_' + input_image_name.split('.')[0])
      print("> Created heatmap original vs blurred")

    # Blurring: multiplication in frequency domain
    # TODO
    #dft_blurred_image = dft_image_array * dft_kernel_blur
    #save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image_f)
    #print("> Created DFT blurred in Fourier " + dft_blur_image_f)
    #blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
    #save_pgm_file(blurred_image_array[0:width, 0:height], blurred_image_f)
    #print("> Created blurred in Fourier " + blurred_image_f)
    #print("SNR after blurring in Fourier: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")

    blur_snr = compute_SNR_in_dB(grayscale_image_array, blurred_image_array)
    print("SNR after blurring: " + str(blur_snr) + " dB")

    if(not no_symm):
      dft_blurred_image = np.fft.fft2(symmetrization(blurred_image_array))
    else:
      dft_blurred_image = np.fft.fft2(blurred_image_array)

    if(not no_images):
      save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image)
      print("> Created abs log scaled DFT blurred " + dft_blur_image)
    
    # Inverse kernel
    if(kernel_size_2 != kernel_size):
      kernel_blur = gaussian_blur_kernel(kernel_size_2)
    if(not no_symm):
      zero_padded_kernel = zero_pad(kernel_blur, symm_image_array.shape[0], symm_image_array.shape[1])
    else:
      zero_padded_kernel = zero_pad(kernel_blur, grayscale_image_array.shape[0], grayscale_image_array.shape[1])
    dft_kernel_blur = np.fft.fft2(zero_padded_kernel)
    if(not no_images):
      save_pgm_file(rearrange_FFT(dft_kernel_blur), dft_kernel_image)
      print("> Created abs log scaled DFT kernel " + dft_kernel_image)
   
    if(plot):
      limits = np.arange(0.0+plot_step, 1.0, plot_step)
      snrs = []
      for limit in limits:
        print("Iteration " + str(limit))
        inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
        for j in range(inverse_dft_kernel_blur.shape[0]):
            for i in range(inverse_dft_kernel_blur.shape[1]):
                if np.abs(dft_kernel_blur[j][i]) <= limit:
                    inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
                else:
                    inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
        
        deconv_f = dft_blurred_image * inverse_dft_kernel_blur
        deconv_s = np.fft.ifft2(deconv_f)
        if(not no_symm):
          deconv_s_crop = deconv_s.real.real[:width, :height]
        else:
          deconv_s_crop = deconv_s.real
        # TODO separate function Clip
        for y in range(height):
          for x in range(width):
            if(deconv_s_crop[y][x] < 0):
              deconv_s_crop[y][x] = 0
            if(deconv_s_crop[y][x] > 255):
              deconv_s_crop[y][x] = 255
        snr = compute_SNR_in_dB(grayscale_image_array, deconv_s_crop.astype(np.uint8))
        snrs.append(snr - blur_snr)
      # For each kernel size, append the SNRs
      snrs_kernel.append(snrs)
    
    inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
    for j in range(inverse_dft_kernel_blur.shape[0]):
        for i in range(inverse_dft_kernel_blur.shape[1]):
            if np.abs(dft_kernel_blur[j][i]) <= threshold:
                inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
            else:
                inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
            #print(np.abs(inverse_dft_kernel_blur[j][i]))
    #save_pgm_file(rearrange_FFT(inverse_dft_kernel_blur), inverse_kernel_image)
    
    deconv_f = dft_blurred_image * inverse_dft_kernel_blur
    if(not no_images):
      save_pgm_file(rearrange_FFT(deconv_f), dft_deconv_image)
      print("> Created abs log scaled DFT de-blurred " + dft_deconv_image)
    deconv_s = np.fft.ifft2(deconv_f)
    if(not no_symm):
      deconv_s_crop = deconv_s.real.real[:width, :height]
    else:
      deconv_s_crop = deconv_s.real
    # Clip
    for y in range(height):
      for x in range(width):
        if(deconv_s_crop[y][x] < 0):
          deconv_s_crop[y][x] = 0
        if(deconv_s_crop[y][x] > 255):
          deconv_s_crop[y][x] = 255
    snr = compute_SNR_in_dB(grayscale_image_array, deconv_s_crop.astype(np.uint8))
    print("SNR after deblurring: " + str(snr) + " dB => Recovered " + str(snr - blur_snr) + " dB.")

    if(not no_images):
      save_pgm_file(deconv_s_crop.astype(np.uint8), deconv_image)
      print("> Created de-blurred image " + deconv_image)

      diff_image_array = grayscale_image_array.astype(np.int16) - deconv_s_crop.astype(np.int16)
      diff_image_array = np.abs(diff_image_array).astype(np.uint8)
      heatmap(diff_image_array, 'heatmap_deconv_' + str(kernel_size) + '_' + str(kernel_size_2) + '_' + input_image_name.split('.')[0])
      print("> Created heatmap original vs de-blurred")

  # For each image append the SNRS of each kernel size
  snrs_kernel_image.append(snrs_kernel)

snrs_kernel_image = np.array(snrs_kernel_image)

if(plot):
  legend = input_image_name_arr
  for i, k in enumerate(kernel_sizes):
    plot_snr(limits, snrs_kernel_image[:,i], k, kernel_sizes_2[i], legend)

