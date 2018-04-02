# Löwner-John Ellipse
This code finds the smallest ellipse covering a finite set of points using [Welzl's algorithm](https://link.springer.com/chapter/10.1007/BFb0038202). This is a recursive algorithm that computes the minimal ellipse containing a set of points P in the interior and another set of points R on the boundary. At each call, the function removes a random point p from P, then recursively computes the smallest ellipse for P - {p}. It then checks if p is inside the ellipse; if it is, then this ellipse is also the minimal one for P. If it isn't, then p must be on the boundary, and we recursively call the function again to compute the smallest ellipse for interior set P - {p} and boundary set R ∪ {p}.

The base cases of the algorithm consist of computing the smallest ellipse when there are 5, 4, or 3 points on the boundary and no points in the interior (the cases of 2 and 1 boundary points are degenerate).
An ellipse is uniquely defined by 5 points on the boundary, therefore this case is easily handled by solving a system of linear equations.  
Additionally, the case of 3 points {p<sub>1</sub>,p<sub>2</sub>,p<sub>3</sub>} is known to be given by (x - c)<sup>T</sup> F (x - c) = 1,
where c = (p<sub>1</sub> + p<sub>2</sub> + p<sub>3</sub>) / 3 and 
F<sup>-1</sup> = 2[(p<sub>1</sub> - c)(p<sub>1</sub> - c)<sup>T</sup> + (p<sub>2</sub> - c)(p<sub>2</sub> - c)<sup>T</sup> + (p<sub>3</sub> - c)(p<sub>3</sub> - c)<sup>T</sup>] / 3.  
The difficult case is 4 boundary points, which is found by the algorithm described in  
[B. W. Silverman and D. M. Titterington. "Minimum covering ellipses." SIAM Journal on Scientific and Statistical Computing 1, no. 4 (1980): 401-409.](https://epubs.siam.org/doi/abs/10.1137/0901028?journalCode=sijcd4)
