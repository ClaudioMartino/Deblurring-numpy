# Deblurring-numpy

Buongiorno a tutti,

La presente repository ha puramente scopi educativi. Consiste in una semplice implementazione in [Python3](https://www.python.org/) di un algoritmo di [deconvoluzione](https://en.wikipedia.org/wiki/Deconvolution) per recuperare un'immagine corrotta da una sfocatura gaussiana ("[deblurring](https://en.wikipedia.org/wiki/Deblurring)"). Siete liberi di utilizzare il codice come meglio credete, citate l'autore originale (io) in caso di pubblicazione e nel caso vi fosse utile fatemelo sapere.

## Prerequisiti
- Python 3.11.1
- Numpy 1.23.0

Ho cercato di limitare al minimo le librerie esterne, solo Numpy è necessario (semplicità di gestione di array 2D e presenza di FFT), ma non escludo un giorno di creare uno script totalmente indipendente. Numpy può essere installato con `pip3 install numpy`.

## Teoria
La trasformata di Fourier discreta (DFT) di un segnale 2D $f(x,y)$ (un'immagine, per esempio) con dimensioni $N \times M$ è:

$$ F(u,v) = \mathcal{F} [ f(x,y) ] = \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} f(x,y) exp[-j 2 \pi (\frac{ux}{M} + \frac{vy}{N}) ] $$

Nel caso di un'immagine quadrata $N \times N$ diventa:

$$ F(u,v) = \mathcal{F} [ f(x,y) ] = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) exp[-j 2 \pi \frac{ux + vy}{N} ] $$

La convoluzione 2D è:

$$ g(x,y)= \omega(x,y) * f(x,y) = \sum_{dx=-\infty}^\infty \sum_{dy=-\infty}^\infty \omega (dx,dy)f(x-dx,y-dy) $$

$$ g(x,y)= f(x,y) * \omega(x,y) = \sum_{dx=-\infty}^\infty \sum_{dy=-\infty}^\infty \omega (x-dx,y-dy)f(dx,dy) $$

Un filtro gaussiano 3x3 consiste in una convoluzione dove $\omega(x,y)$ è la matrice (il "[kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing))") definita come

$$
\frac{1}{16}
\begin{bmatrix}
\ \ 1 &\ \  2 &\ \  1 \\
\ \ 2 &\ \  4 &\ \  2 \\
\ \ 1 &\ \  2 &\ \  1
\end{bmatrix}
$$

Nel dominio delle frequenze, la convoluzione corrisponde al prodotto, quindi:

$$ G(u,v) = F(u,v) \Omega(u,v) $$

Di conseguenza è possibile recuperare il segnale originale calcolando

$$ F(u,v) = \frac{G(u,v)}{\Omega(u,v)} $$

e prendendo la trasformata di Fourier inversa. 

> [!NOTE]
> I precedenti calcoli non tengono in considerazione il rumore introdotto ad ogni passaggio. In quel caso bisognerebbe stimare  $\hat{f}(x,y)$ che minimizza l'errore quadratico medio $\mathbb{E} \left| f(x,y) - \hat{f}(x,y) \right|^2$ (v. [Wiener](https://en.wikipedia.org/wiki/Wiener_deconvolution)).

## Struttura 
Come input abbiamo un'immagine RGB con estensione [`.ppm`](https://en.wikipedia.org/wiki/Netpbm). Il file viene immediatamente convertito in scala di grigi e salvato come `.pgm`. Nella repository sono già presenti varie immagini di esempio $512 \times 512$, tra cui la classica [Lenna](https://en.wikipedia.org/wiki/Lenna)[^1] e la meno classica (ma ben più importante) Sabrina Salerno[^2].

L'immagine in bianco e nero viene filtrata in modo da ottenerne una versione sfocata. Il filtro di Gauss è implementato con una convoluzione 2D o con un prodotto nel dominio delle frequenze:
- nel primo caso la funzione che esegue la convoluzione è stata scritta a mano;
- nel secondo caso si è utilizzata la FFT 2D di Numpy ([`fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)) per calcolare $F(u,v)$ e $\Omega(u,v)$.

## Risultati
Di seguito i risultati ottenuti con Lena (qui le immagini sono riscalate e convertite in `.png`). L'immagine di input viene convertita in scala di grigi.

<p>
<a href="docs/images/lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/lena.png" width="300"></a>
<a href="docs/images/lena_gray.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/lena_gray.png" width="300"></a>
</p>

L'immagine è stata sfocata con una convoluzione (3x3, 5x5, 7x7). L'operazione di de-convoluzione restituisce un'immagine più nitida, attenuando gli effetti della sfocatura.

<p>
<a href="docs/images/3_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/3_lena_blur.png" width="300"></a>
<a href="docs/images/3_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/3_lena_deconv.png" width="300"></a>
</p>

<p>
<a href="docs/images/5_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/5_lena_blur.png" width="300"></a>
<a href="docs/images/5_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/5_lena_deconv.png" width="300"></a>
</p>

<p>
<a href="docs/images/7_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/7_lena_blur.png" width="300"></a>
<a href="docs/images/7_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/7_lena_deconv.png" width="300"></a>
</p>

## Risultati

<a href="images/results/plot_3.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/images/results/plot_3.png"></a>

<a href="images/results/plot_5.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/images/results/plot_5.png"></a>

<a href="images/results/plot_7.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/images/results/plot_7.png"></a>

## Fonti
[^1]: *Playboy*, vol. 19, [n. 11](https://images4.imagebam.com/cd/10/16/ME12BMO_o.jpg), novembre 1972, Playboy Enterprises.
[^2]: *Playmen*, anno XXII, [n. 9](https://images3.imagebam.com/8e/1b/c6/54f4fd195104304.jpg), settembre 1988, Tattilo Editrice.
