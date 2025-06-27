#include "param.h"
#include "types.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "defs.h"
#include "proc.h"

/* NTU OS 2024 */
/* Allocate eight consecutive disk blocks. */
/* Save the content of the physical page in the pte */
/* to the disk blocks and save the block-id into the */
/* pte. */
char *swap_page_from_pte(pte_t *pte) {
  char *pa = (char*) PTE2PA(*pte);
  uint dp = balloc_page(ROOTDEV);

  write_page_to_disk(ROOTDEV, pa, dp); // write this page to disk
  *pte = (BLOCKNO2PTE(dp) | PTE_FLAGS(*pte) | PTE_S) & ~PTE_V; // remove dirty bit

  return pa;
}

/* NTU OS 2024 */
/* Page fault handler */
int handle_pgfault() {
  /* Find the address that caused the fault */
  /* uint64 va = r_stval(); */

  /* TODO */
  uint64 va = r_stval();
  va = PGROUNDDOWN(va);

  struct proc *p = myproc();

  pte_t *pte = walk(p->pagetable, va, 0);
  uint64 blockno = PTE2BLOCKNO(*pte);

  if (pte == 0) {
    panic("handle_pgfault: pte should exist");
  }
  if (*pte & PTE_V) {
    panic("handle_pgfault: pte should not be valid");
  }

  uint64 pa = (uint64) kalloc();
  memset((void *)pa, 0, PGSIZE);
  int perm = PTE_U | PTE_R | PTE_W | PTE_X;

  mappages(myproc()->pagetable, va, PGSIZE, pa, perm);

  read_page_from_disk(ROOTDEV, pa, blockno);
  bfree_page(ROOTDEV, blockno);

  return 0;
}
