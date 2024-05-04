# Deblurring-numpy

La presente repository ha puramente scopi educativi. Consiste in una semplice implementazione in [Python3](https://www.python.org/) di un algoritmo di [deconvoluzione](https://en.wikipedia.org/wiki/Deconvolution) per recuperare un'immagine corrotta da una sfocatura gaussiana ("[deblurring](https://en.wikipedia.org/wiki/Deblurring)"). Siete liberi di utilizzare il codice come meglio credete, citate l'autore originale (io) in caso di pubblicazione e nel caso vi fosse utile fatemelo sapere.

## Prerequisiti
- Python 3.11.1
- Numpy 1.23.0
- Matplotlib 3.8.4

Ho cercato di limitare al minimo le librerie esterne, solo Numpy è necessario (semplicità di gestione di array 2D e presenza di DFT), ma non escludo un giorno di creare uno script totalmente indipendente. Numpy può essere installato con `pip3 install numpy`. Se volete salvare dei grafici anche Matplotlib deve essere installato con `pip3 install matplotlib`.

## Cenni teorici
La trasformata di Fourier discreta (DFT) di un segnale 2D $f(x,y)$ (un'immagine, per esempio) con dimensioni $N \times M$ è:

$$ F(u,v) = \mathcal{F} [ f(x,y) ] = \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} f(x,y) exp[-j 2 \pi (\frac{ux}{M} + \frac{vy}{N}) ] $$

La convoluzione discreta 2D è:

$$ g(x,y)= \omega(x,y) * f(x,y) = \sum_{dx=-\infty}^\infty \sum_{dy=-\infty}^\infty \omega (dx,dy)f(x-dx,y-dy) $$

Un filtro gaussiano consiste in una convoluzione 2D tra un'immagine $f(x,y)$ e una matrice $\omega(x,y)$ che rappresenta una curva normale discreta (il "[kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing))"). Il kernel 3x3, per esempio, è definito come

$$
\frac{1}{16}
\begin{bmatrix}
\ \ 1 &\ \  2 &\ \  1 \\
\ \ 2 &\ \  4 &\ \  2 \\
\ \ 1 &\ \  2 &\ \  1
\end{bmatrix}
$$

Nel dominio delle frequenze:

$$ G(u,v) = \mathcal{F} [ g(x,y) ] = \mathcal{F} [ f(x,y) * \omega(x,y) ] = F(u,v) \Omega(u,v) $$

dove $\Omega(u,v) = \mathcal{F} [ \omega(x,y) ]$ è la DFT del kernel. Di conseguenza è possibile recuperare il segnale originale calcolando

$$ f(x, y) = \mathcal{F}^{-1} [ F(u,v) ] = \mathcal{F}^{-1} \left[ \frac{G(u,v)}{\Omega(u,v)} \right] $$

dove $\mathcal{F}^{-1}$ è la trasformata di Fourier discreta inversa (IDFT).

> [!NOTE]
> I precedenti calcoli non tengono in considerazione il rumore introdotto ad ogni passaggio. In quel caso bisognerebbe stimare $\hat{f}(x,y)$ che minimizza l'errore quadratico medio $\mathbb{E} \left| f(x,y) - \hat{f}(x,y) \right|^2$ (v. [deconvoluzione di Wiener](https://en.wikipedia.org/wiki/Wiener_deconvolution)).

## Esperimento 1
### Descrizione
Come input è stata usata un'immagine RGB con estensione [`.ppm`](https://en.wikipedia.org/wiki/Netpbm). Il file viene convertito in scala di grigi e salvato come `.pgm`. Nella repository sono già presenti varie immagini di esempio $512 \times 512$, tra cui la classica [Lenna](https://en.wikipedia.org/wiki/Lenna)[^1] e la meno classica Sabrina Salerno[^2]. Altre immagini .ppm possono essere utilizzate, basta aggiungerle alla medesima cartella.

<p>
<a href="docs/images/lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/lena.png" width="300"></a>
<a href="docs/images/lena_gray.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/lena_gray.png" width="300"></a>
</p>

L'immagine in bianco e nero viene sfocata tramite convoluzione. L'energia presente nell'immagine sfocata è confrontata con quella dell'immagine originale e viene calcolato l'SNR. Viene calcolato l'inverso della trasformata di Fourier del kernel scelto (a sua volta un'altra curva gaussiana) che viene poi moltiplicato per lo spettro dell'immagine sfocata. Del risultato viene conservata solo la parte reale, dato che la parte immaginaria conterrà solo rumore. I valori superiori a 255 e quelli inferiori a 0 vengono tagliati e il tutto viene convertito in byte. Viene calcolato anche l'SNR dell'immagine risultante e confrontato con quello dell'immagine sfocata.

### Problemi
#### Problema 1
I bordi...

#### Problema 2
La Symmetrization dell'input è fondamentale per evitare di introdurre alte frequenze e invalidare ogni misura.

#### Problema 3
Le alte frequenze del kernel sono nulle o prossime allo zero e invertirle dà origine a valori molto elevati. Moltiplicare questi valori per le alte frequenze dell'immagine sfocata porterà all'introduzione di rumore. Di conseguenza è stata definita una soglia sul valore assoluto oltre la quale le frequenze non sono state invertite, ma portate a 1, al fine di non modificare le alte frequenze dell'immagine, comunque già di per sè poco rilevanti in un'immagine naturale. Un processo iterativo a portato a definire 0,07 come valore di soglia medio grazie al quale è stato possibile recuperare circa 2 dB di SNR dalle immagini usate nel test. Ogni immagine avrà, naturalmente, un preciso valore in corrispondenza del quale si potrà recuperare il massimo di dB di SNR.

<p>
<a href="docs/images/plot_3_3.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_3_3.png" width="250"></a>
<a href="docs/images/plot_5_5.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_5_5.png" width="250"></a>
<a href="docs/images/plot_7_7.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_7_7.png" width="250"></a>
</p>

### Risultati
Di seguito si riportano i risultati ottenuti con Lena (qui le immagini sono convertite in `.png`).

L'immagine è stata sfocata con una convoluzione (3x3, 5x5, 7x7):

<p>
<a href="docs/images/3_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/3_lena_blur.png" width="300"></a>
<a href="docs/images/heatmap_blur_3_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_blur_3_lena.png" width="300"></a>
</p>

<p>
<a href="docs/images/5_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/5_lena_blur.png" width="300"></a>
<a href="docs/images/heatmap_blur_5_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_blur_5_lena.png" width="300"></a>
</p>

<p>
<a href="docs/images/7_lena_blur.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/7_lena_blur.png" width="300"></a>
<a href="docs/images/heatmap_blur_7_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_blur_7_lena.png" width="300"></a>
</p>

La de-convoluzione (soglia: 0,07) ha restituito le seguenti immagini:

<p>
<a href="docs/images/3_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/3_lena_deconv.png" width="300"></a>
<a href="docs/images/heatmap_deconv_3_3_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_deconv_3_3_lena.png" width="300"></a>
</p>

<p>
<a href="docs/images/5_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/5_lena_deconv.png" width="300"></a>
<a href="docs/images/heatmap_deconv_5_5_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_deconv_5_5_lena.png" width="300"></a>
</p>

<p>
<a href="docs/images/7_lena_deconv.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/7_lena_deconv.png" width="300"></a>
<a href="docs/images/heatmap_deconv_7_7_lena.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/heatmap_deconv_7_7_lena.png" width="300"></a>
</p>

<table>
  <tr>
    <th rowspan="3">File</th>
    <th colspan="9">SNRs [dB]</th>
  </tr>
  <tr>
    <th colspan="3">3x3</th>
    <th colspan="3">5x5</th>
    <th colspan="3">7x7</th>
  </tr>
  <tr>
    <th>Blurred</th>
    <th>De-blurred</th>
    <th>Difference</th>
    <th>Blurred</th>
    <th>De-blurred</th>
    <th>Difference</th>
    <th>Blurred</th>
    <th>De-blurred</th>
    <th>Difference</th>
  </tr>
  <tr>
<td>sabrina.ppm</td>

<td>9.42 </td>
<td>12.18 </td>
<td>2.77 </td>

<td>6.99 </td>
<td>10.26 </td>
<td>3.27 </td>

<td>6.92 </td>
<td>10.03 </td>
<td>3.11 </td>
  </tr>
  <tr>
<td>lena.ppm</td>

<td>6.41 </td>
<td>9.70 </td>
<td>3.29 </td>

<td>4.25 </td>
<td>7.76 </td>
<td>3.50 </td>

<td>4.31 </td>
<td>7.91 </td>
<td>3.60 </td>
  </tr>
  <tr>
<td>sara.ppm</td>

<td>5.17 </td>
<td>10.55 </td>
<td>5.38 </td>

<td>2.20 </td>
<td>7.07 </td>
<td>4.88 </td>

<td>2.02 </td>
<td>6.52 </td>
<td>4.50 </td>
  </tr>
  <tr>
<td>laura.ppm</td>

<td>5.89 </td>
<td>9.07 </td>
<td>3.18 </td>

<td>3.84 </td>
<td>6.88 </td>
<td>3.04 </td>

<td>3.84 </td>
<td>6.95 </td>
<td>3.11 </td>
  </tr>
</table>

## Esperimento 2
Che succede se si prova a de-sfocare un'immagine con l'inverso di un kernel diverso da quello usato per sfocarla?

<p>
<a href="docs/images/plot_3_5.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_3_5.png" width="300"></a>
<a href="docs/images/plot_3_7.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_3_7.png" width="300"></a>
</p>

<p>
<a href="docs/images/plot_5_3.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_5_3.png" width="300"></a>
<a href="docs/images/plot_5_7.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_5_7.png" width="300"></a>
</p>

<p>
<a href="images/results/plot_7_3.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_7_3.png" width="300"></a>
<a href="images/results/plot_7_5.png"><img src="https://raw.githubusercontent.com/ClaudioMartino/Deblurring-numpy/main/docs/images/plot_7_5.png" width="300"></a>
</p>

Risulta evidente che la conoscenza del filtro di sfocatura è fondamentale per recuperare al meglio l'immagine originale. Tuttavia il de-blurring con dimensioni inferiori a quelle della sfocatura ha comunque permesso di recuperare qualche dB. Inoltre i filtri 5x5 e 7x7 paiono molto simili.

[^1]: *Playboy*, vol. 19, [n. 11](https://images4.imagebam.com/cd/10/16/ME12BMO_o.jpg), novembre 1972, Playboy Enterprises.
[^2]: *Playmen*, anno XXII, [n. 9](https://images3.imagebam.com/8e/1b/c6/54f4fd195104304.jpg), settembre 1988, Tattilo Editrice.
