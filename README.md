# Deblurring-numpy

Buongiorno a tutti,

La presente repository ha puramente scopi educativi. Consiste in una semplice implementazione in [Python3](https://www.python.org/) di un algoritmo di [deconvoluzione](https://en.wikipedia.org/wiki/Deconvolution) per recuperare un'immagine corrotta da una sfocatura gaussiana ("[deblurring](https://en.wikipedia.org/wiki/Deblurring)"). Siete liberi di utilizzare il codice come meglio credete, citate l'autore originale (io) in caso di pubblicazione e nel caso vi fosse utile fatemelo sapere.

## Prerequisiti
- Python 3.8.2
- Numpy 1.23.0

Numpy può essere installato con `pip install numpy`.

## Teoria
La trasformata di Fourier di un segnale 2D $f(x,y)$ con dimensioni $N \times N$ (un'immagine, per esempio) è:

$$ F(u,v) = \mathcal{F} [ f(x,y) ] = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) e^{-j 2 \pi (ux + vy)/N } $$

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

I precedenti calcoli non tengono in considerazione il rumore introdotto ad ogni passaggio. In quel caso bisognerebbe stimare  $\hat{f}(x,y)$ che minimizza l'errore quadratico medio $\mathbb{E} \left| f(x,y) - \hat{f}(x,y) \right|^2$ (v. [Wiener](https://en.wikipedia.org/wiki/Wiener_deconvolution)).

## Struttura 
Come input abbiamo un'immagine RGB con estensione [`.ppm`](https://en.wikipedia.org/wiki/Netpbm). Il file viene immediatamente convertito in scala di grigi e salvato come `.pgm`. Nella repository sono già presenti due immagini di esempio: la classica [Lenna](https://en.wikipedia.org/wiki/Lenna) [1] e la meno classica Sabrina Salerno [2].

## Risultati
[<img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/sabrina.png" width="200">](docs/images/sabrina.png)

[<img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/sabrina_gray.png" width="200">](docs/images/sabrina_gray.png)

## Fonti
- [1] *Playboy*, novembre 1972, Playboy Enterprises
- [2] *Playmen*, settembre 1988, Tattilo Editrice
