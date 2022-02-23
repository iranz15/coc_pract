
static char help[] = "Reads a PETSc matrix from a file and solves a linear system.\n\n";

#include <petscksp.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;               /* matrix */
  Vec            x,b,r;           /* approx solution, RHS, aux vector */
  PetscViewer    vw;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Load the matrix */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vw);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  /* Create vectors */
  ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Setup and solve system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* COMPLETE: Check residual */

  /* Free work space */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

