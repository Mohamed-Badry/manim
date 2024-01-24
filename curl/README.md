The code was used to make [this video](https://youtu.be/fwytVQdm3iI).

### Some Notes
- Manim isn't included in the `requirements.txt` file as it requires a different installation method provided [here](https://docs.manim.community/en/stable/installation.html).
- The Curl class is where you can input your 2D vector function.
- The vector function has to use sympy functions for example sym.sin(), sym.cos(), and so on.
- The variables have to belong to the $R$ coordinate system so $[R.x, R.y]$ have to be used instead of $[x, y]$, the same applies to the coordinate directions $[R.i, R.j]$.
