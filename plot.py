import matplotlib.pyplot as plt

def plot_snr(l, snrss, k, k2, legend):
  fig, ax = plt.subplots()

  plt.grid(axis='x', color='0.95')
  plt.grid(axis='y', color='0.95')
  
  plt.xscale('log')
  
  #plt.hlines(blur_l, l[0], l[-1], color='C0', linestyle='dashed')
  #plt.hlines(blur_s, l[0], l[-1], color='C1', linestyle='dashed')
  
  #plt.plot(l, snr_l, color='C0')
  #plt.plot(l, snrs, color='C1')
  for snrs in snrss:
    plt.plot(l, snrs)
  
  #plt.legend(('Lena blurred', 'Sabrina blurred', 'Lena de-blurred', 'Sabrina de-blurred'))
  plt.legend(legend)
  
  plt.title('blurring = ' + str(k) + ", deblurring = " + str(k2))

  plt.xlabel('Threshold')
  plt.ylabel('Normalized SNR [dB]')

  plt.savefig('images/results/plot_' + str(k) + '_' + str(k2) + '.png', bbox_inches='tight')
  #plt.show()

def heatmap(array, image):
  fig, ax = plt.subplots()
  im = ax.imshow(array, cmap='plasma')
  cbar = ax.figure.colorbar(im, ax=ax)
  plt.axis('off')

  plt.savefig('images/results/' + image + '.png', bbox_inches='tight')
  #plt.show()

