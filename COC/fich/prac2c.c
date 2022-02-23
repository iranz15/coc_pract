
static char help[] = "Tests MPI parallel matrix creation.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A;
  PetscInt       i,j,k,m=20,n=20,Istart,Iend;
  PetscErrorCode ierr;
  Vec            v,w;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* COMPLETE: load n and m from the command line */

  /* COMPLETE: create matrix A, must have m*n rows and m*n columns */

  /* Fill the matrix for the five point stencil */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (k=Istart;k<Iend;k++) {
    i = k/n; j = k-i*n;
    ierr = MatSetValue(A,k,k,4.0,INSERT_VALUES);CHKERRQ(ierr);
    if (i>0) { ierr = MatSetValue(A,k,k-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A,k,k+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,k,k-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(A,k,k+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* COMPLETE: matrix assembly */

  /* COMPLETE: create vectors v,w, fill v with all 1's, and compute the product w=A*v */

  /* COMPLETE: show vector w */

  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
