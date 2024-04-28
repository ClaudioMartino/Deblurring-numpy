import matplotlib.pyplot as plt

def plot_snr(l, snrss, k, legend):
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
  
  plt.title('kernel size = ' + str(k))

  plt.xlabel('Threshold')
  plt.ylabel('Normalized SNR [dB]')

  plt.savefig('images/results/plot_' + str(k) + '.png', bbox_inches='tight')
  #plt.show()
