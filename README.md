coucou  
coucou EMMA  
salut adrien   
ça marche  
sur blocnote  
# Quickstart

Dans un dossier de travail faire

```
git clone git@github.com:secretpenguin75/vfcdevel.git
```

Ensuite creer un fichier jupyter notebook et importer la fonction carotte avec

```
from vfcdevel.profilegen import Profile_gen
```

Et ensuite, appeler les fonctions

```
# load your favourite model output
df = pd.read_csv('./vfcdevel/data/lmdz_DC.csv',index_col=0,parse_dates=True)
#df['precipd18O']-=4.1
# create a virtual firn core from the model
VFC = Profile_gen(df['decimalyear'],df['tsol'],df['tp'],df['precipd18O'],320)
plt.plot(VFC['d18O_diff'])

```
also available for era5
````
df = pd.read_csv('./vfcdevel/data/era5.csv',index_col=0,parse_dates=True)
VFC = Profile_gen2(df['decimalyear'],df['t2m'],df['tp'],df['t2m_d18O'],320)
plt.plot(VFC['d18O_diff'])
````
Now you can compare it to your _real_ core

```
ICORDA = pd.read_excel("./data_emma/ICORDA_depth_age_d18O_model_period.xlsx")
df = pd.read_csv('./vfcdevel/data/lmdz_DC.csv',index_col=0,parse_dates = True)
plt.plot(ICORDA['d18O'])
plt.plot(VFC['d18O_diff'])
```
Or use the super function superplot to explore parameter values
```
from vfcdevel.runvfc_v3 import superplot
superplot(df,ICORDA)

```
Pour générer la carotte paleo, simplement utiliser le dataframe paleo en rajoutant une colonne 'tp_adjust' avec l'accu.
Je mets accu = 25kg, 219K, 320kg/m3 en surface et j'obtiens une profondeur de 84m!
```
Tmean =219
rho = 320
accu = 25
dfpaleo['tp'] = accu/12
dfpaleo['decimalyear'] = list(map(str_to_decimalyear,dfpaleo.index))
```
```
vfc = Profile_gen2(dfpaleo['decimalyear'],Tmean,dfpaleo['tp'],dfpaleo['d18O_inter']+np.nanmean(dficorda['d18O']),rho,res=1e-2)
vfc = core_sample_adrien(vfc,core_resolution)
```
etc...

# NEW! in version 3/08

j'ai renomé des fichiers. Normalement j'ai bien fait les changements nécessaires partout
- make_pretty_figures.py -> pretty_plot.py (keep file names short :-))
- psd.py-> spectralanalysis.py (je me dis qu'on pourra rajouter les ondelettes dedans par après)

- Nouvelle version de Profile_gen dans profilegen_v3 (VFC_and_spectra_v2 renomé en VFC_and_spectra dans ce fichier)
how to use:
```
from vfcdevel.profilegen_v3 import Profile_gen,VFC_and_spectra
from runvfc_v3 import superplot
```
Note: On peut toujours faire
```
from vfcdevel.profilegen_v2 import Profile_gen as Profile_gen_v2
from vfcdevel.profilegen_V2 import VFC_and_spectra_v2
from runvfc import supersplot as superplot_v2
```
et comparer...

- J'ai modifié une seule chose dans v2: dans la partie mixing, dans le rolling je me suis permi de rajouter `,min_periods=1` dans le rolling. Garder en tête que le rolling va de gauche à droite/ de haut en bas pour notre profil en profondeur. La version par défaut fait la moyenne des n valeurs à droite (longueur de rolling window) Tu as utilisé "center", qui fait la moyenne des n valeurs autour du point, ce qui fait sens si on imagine qu'une valeur à une certaine profondeur est mélangée avec la neige qui viendra par la suite. Le comportement par défaut de pandas est de mettre des nans si il ne trouve pas ses n valeurs. ce sera le cas tout en haut de la carotte (pas de valeurs à gauche/ au dessus de depth = 0), hors on a bien de la neige, elle est juste uniquement mélangée avec la neige en dessous. min_periods = 1 (au lieu de min_periods = n) permet de ne pas avoir de nan. On pourrait dire min_periods = n//2 mais c'est plus simple de mettre n=1 et le résultat est le même.

Donc, avant
```
vfc_even['d18O_noise'].rolling(window=mixing_scale, center=True).mean().to_numpy()
```
après
```
vfc_even['d18O_noise'].rolling(window=mixing_scale, center=True,min_periods=1).mean().to_numpy()
```

- J'ai également modifié une chose importante au tout début du script. Quand on passe de depth "raw" (la cumsum des precips) à depth even (dz régulier), on utilisait une interpolation pour calculer d18O et compagnie. C'est une erreur que j'avais faite au début en voulant simplifier le code de Mathieu (enlever les boucles for). Je n'arrive pas à remettre la main sur sa version matlab, mais il faisait une sorte d'équivalent de bloc average. C'est plus correcte car si tu y pense, en interpolant on peut facilement tomber entre deux valeurs qui sont très bruitées et qui n'ont rien à voir avec la valeur moyenne du milimetre (ce qui nous intéresse vraiment).
J'avais corrigé ça dans mes versions ultérieures où j'ai ajouté la sublimation etc... mais je n'étais pas revenu sur Profile_gen. 
J'ai remarqué ça car j'avais des grosse différences quand j'ai voulu changer la résolution pour générer la vfc (cm au lieu de milimetre) et quand j'ai remis le bloc average à la place de l'interpolation je n'avais plus ce problème. Pour pouvoir comparer nos version v2 et v3 côte à côte je me suis permi de modifier ça dans le v2 également. Normalement ça va contribuer légèrement à baisser les hautes fréquences. Malheureusement ça ralenti un peu l'execution de la version v2.


### Et maintenant pour les choses cool

- NEW! le v3 sort un dataframe de VFC, donc
```
vfc = Profile_gen(...)
```
et puis simplement
```
plt.plot(vfc[['d18O','d18O_diff']])
```
et oui, c'est merveilleux... Les autres colonnes du vfc sont ['d18O_raw', 'date', 'd18O_noise', 'd18O_mix', 'd18O', 'rho', 'depth_we',
       'd18O_diff', 'sigma18'] ou j'ai gardé les d18O de chaque étapes, ainsi que le profil de densité heron et langway, la profondeur eu, le sigma18. Ça permettra de faire les vérifications très rapidement en vérifiant les profils de densités et les longeurs de diffusion par exemple. On pourra aussi rajouter super facilement les autres espèces.


- Pour le v3, j'ai re-séparé le core_resolution du reste du profile_gen. Mon idée c'est: Profile_gen donne une carotte physique, avec une résolution typiquement milimetrique. Et puis on peut venir derrière appliquer le core_resolution (schéma d'échantillonnage). Je préfère toujours limiter le nombre d'arguments d'une fonction, faire des blocs de fonction qui font chacun une tâche, je trouve ça plus lisible. Si tu veux tu peux faire un core_gen (profile_gen_legacy) qui fait profile_gen et puis core_sample dans la foulée. J'ai aussi remarqué que ma première version de bloc average est super lente pour les plus longues carottes. J'ai réécrit un core_sample_adrien qui n'utilise pas bloc_average, et je vais réecrire bloc average avec la même idée. J'ai vérifié que core_sample_emma et core_sample_adrien font exactement le même résultat :)
```
vfc = Profile_gen(...)
#vfc2 = core_sample_emma(vfc,core_resolution)
vfc2 = core_sample_adrien(vfc,core_resolution)
plt.plot(vfc['d18O'])
plt.plot(vfc2['d18O'])
```

Et puis,

- J'ai rajouté la résolution en entrée, res=1e-3. J'ai mis le mixing scale et noise scale en metre. Ils sont converti à la résolution du vfc avec
```
mixing_scale = mixing_scale_m/res
etc...
```
Ça sera utile pour faire la carotte de 80m en cm au lieu de mm pour faire tourner le code un peu plus vite.

- J'ai homogénéisé un peu partout les noms des entrées et l'ordre dans lequel elle apparaissent: noise_level, mixing_level mixing_scale_m, noise_scale_m, . Ce serait bien on essaye de garder les mêmes noms partout.


# NEW!! August 19th

- Pour utiliser les wavelets
```
from vfcdevel.spectralanalysis import wavelets_adrien
```

```
vfc = Profile_gen2(dfpaleo['decimalyear'],Tmean,dfpaleo['tp'],dfpaleo['d18O_inter']+np.nanmean(dficorda['d18O']),rho,res=1e-3)
vfc = core_sample_adrien(vfc,core_resolution)
vfc2 = df_interp(vfc,np.arange(min(vfc.index),max(vfc.index),0.01))
```
et puis
```
frequencies,coefficients = wavelets_adrien(vfc2['d18O'])
fig,ax = plt.subplots()
plt.pcolormesh(vfc2.index.values,frequencies,np.abs(coefficients),vmin=0,vmax=5,rasterized=True)
plt.gca().set_yscale('log')
```
La fonction que tu m'a envoyée est aussi disponible pour comparer, avec
```
from vfcdevel.spectralanalysis import wavelets_fft_spectra

wavelets_fft_spectra(vfc2,'ICORDA')

```
Je pense que l'axe y n'est pas correcte dans cette version, voir
```
wavelets_fft_spectra(vfc2,'ICORDA',newversion=True)

```


- J'ai rajouté le storage diffusion avec
```
vfc = Profile_gen2(dfpaleo['decimalyear'],Tmean,dfpaleo['tp'],dfpaleo['d18O_inter']+np.nanmean(dficorda['d18O']),rho,res=1e-3,storage_diffusion_cm = 1.5)
```
vfc aura deux colonnes en plus avec 'd18O_diff2' qui rajoute la diffusion du stockage et 'sigma18_storage'. Ça a l'air de faire aucune différence pour icorda. En fait ça change à peine seulement dans le premier 1 mètre.

# update sept 3;
On peut maintenant gérer dexc et d18O en parralèle avec
```
Profile_gen(df['decimalyear'],df['tsol'],df['tp_adjust'],df[['tsol_dexc','tsol_d18O']],320, mixing_level=0., noise_level=0.,storage_diffusion_cm = 3)
```
(NB: les doubles [[ ]] dans la quatrième entrée. L'espèce est assignée automatiquement à d18O ou dexc (ou dD) en fonction de l'intitulé de la colonne. Le choix est printed pour vérification par l'utilisateur)
Par conséquent, storage diffusion désormais être un entier (même diffusion sur les deux espèces d18O et dD) ou un dictionnaire de la forme ```{'d18O':sigma18_storage_cm,'dD':sigmaD_storage_cm}```.

Tout est backward compatible avec une seule espèce, on peut donc aussi bien faire
```
Profile_gen(df['decimalyear'],df['tsol'],df['tp_adjust'],df[['tsol_d18O']],320, mixing_level=0., noise_level=0.,storage_diffusion_cm = 3)
```
(une seule colonne) que
```
Profile_gen(df['decimalyear'],df['tsol'],df['tp_adjust'],df['tsol_d18O'],320, mixing_level=0., noise_level=0.,storage_diffusion_cm = 3)
```
comme avant.

_to be continued..._
