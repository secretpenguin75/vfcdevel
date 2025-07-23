coucou
coucou EMMA

# Quickstart
Dans un dossier de travail faire

```
git clone git@github.com:secretpenguin75/vfc-devel.git
```

Ensuite creer un fichier jupyter notebook et importer la fonction carotte avec

```
from vfcdevel.profilegen import Profile_gen
```

Et ensuite, appeler les fonctions

```
depth,depthwe,rho,iso,iso_diff,date,sigma = Profile_gen(df['year'],df['tp'],df['d18O_inter'],Tmean,350)

```
etc...

