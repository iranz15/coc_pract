
static char help[] = "Tests PetscRandom functions.\n\n";

#include <petscsys.h>

/*
   Based on PETSc example $PETSC_DIR/src/sys/classes/random/tutorials/ex1.c

   Usage:
   ./ex1 -n <num_of_random_numbers> -random_type <type> -log_view -view_values
*/

int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
  PetscRandom    rnd;
  PetscScalar    value,avg = 0.0;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      view_values=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size!=1) SETERRQ(PETSC_COMM_SELF,1,"This program works with one MPI process only");

  /* command-line options: for instance
 
       $ ./prac1a -n 2000

     will set the value of variable n to 2000. If the option is not given the variable
     keeps the value given in the initialization (1000 in this case).
  */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_values",&view_values,NULL);CHKERRQ(ierr);

  /* create a PetscRandom object, set it to generate random numbers in the interval [0,1] */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd,0.0,1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  /* allocate an array of integer values and fill it */
  ierr = PetscPrintf(PETSC_COMM_SELF,"Generating %D random values\n",n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
    avg += value;
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);  /* convert the random number to an integer */
  }
  avg = avg/((PetscReal)n);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Average value %6.4e\n",(double)PetscRealPart(avg));CHKERRQ(ierr);

  /* print generated integers */
  if (view_values) {
    for (i=0; i<n; i++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"values[%D] = %D\n",i,values[i]);CHKERRQ(ierr);
    }
  }

  /* sort the integer values and check they have been sorted correctly */
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (values[i] < values[i-1]) SETERRQ(PETSC_COMM_SELF,1,"Values not sorted");
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"Values sorted correctly\n");CHKERRQ(ierr);

  /* free allocated memory and objects */
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

