
static char help[] = "Orthogonalization of vectors.\n\n";

#include <petscvec.h>

PetscErrorCode CGS(PetscInt k,Vec *v,Vec *q)
{
  PetscErrorCode ierr;
  PetscScalar    *R;
  PetscReal      norm;
  PetscInt       i,j;
  Vec            w;

  PetscFunctionBegin;
  ierr = VecDuplicate(v[0],&w);CHKERRQ(ierr);
  ierr = PetscMalloc1(k*k,&R);CHKERRQ(ierr);
  ierr = PetscArrayzero(R,k*k);CHKERRQ(ierr);
  for (j=0;j<k;j++) {
    ierr = VecCopy(v[j],w);CHKERRQ(ierr);
    for (i=0;i<j;i++) {
      ierr = VecDot(v[j],q[i],&R[i+j*k]);CHKERRQ(ierr);
      ierr = VecAXPY(w,-R[i+j*k],q[i]);CHKERRQ(ierr);
    }
    ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
    R[j+j*k] = norm;
    ierr = VecCopy(w,q[j]);CHKERRQ(ierr);
    ierr = VecScale(q[j],1.0/R[j+j*k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(R);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec            x,*v,*q;
  PetscInt       i,n=20,k=3;
  PetscRandom    rnd;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* COMPLETE: load n and k from the command line */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Orthogonalizing %D vectors of length %D\n",k,n);CHKERRQ(ierr);

  /*
     Create a vector that will be used as a template for all vectors.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /*
     Duplicate some vectors (of the same format and
     partitioning as the initial vector).
  */
  /* COMPLETE: create arrays of vectors v and q */

  /*
     Set the v vectors with random entries.
  */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = VecSetRandom(v[i],rnd);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);

  /*
     Call orthogonalization routine.
  */
  ierr = CGS(k,v,q);CHKERRQ(ierr);

  /*
     Free work space. All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  /* COMPLETE: destroy arrays of vectors v and q */
  ierr = PetscFinalize();
  return ierr;
}

