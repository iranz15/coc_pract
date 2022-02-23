default: ex1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o

prac1a: prac1a.o
	-${CLINKER} -o prac1a  prac1a.o ${PETSC_KSP_LIB}
	${RM} prac1a.o

prac2a: prac2a.o
	-${CLINKER} -o prac2a  prac2a.o ${PETSC_KSP_LIB}
	${RM} prac2a.o

prac2b: prac2b.o
	-${CLINKER} -o prac2b  prac2b.o ${PETSC_KSP_LIB}
	${RM} prac2b.o

prac2b_multi: prac2b_multi.o
	-${CLINKER} -o prac2b_multi  prac2b_multi.o ${PETSC_KSP_LIB}
	${RM} prac2b_multi.o

prac2c: prac2c
	-${CLINKER} -o prac2c  prac2c.o ${PETSC_KSP_LIB}
	${RM} prac2c.o

prac3a: prac3a
	-${CLINKER} -o prac3a  prac3a.o ${PETSC_KSP_LIB}
	${RM} prac3a.o

prac3b: prac3b
	-${CLINKER} -o prac3b  prac3b.o ${PETSC_KSP_LIB}
	${RM} prac3b.o

