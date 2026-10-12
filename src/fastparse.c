/*     CalculiX - A 3-dimensional finite element program                 */
/*     Fast input deck tokenizer and numeric parser                      */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "CalculiX.h"

#define NENTRIES 19

/* Fast replacement for getnewline + splitline:
   Reads directly from inpc[ipoinpc[iline-1] .. ipoinpc[iline]-1] without copying
   into a 1320-char intermediate buffer, and uses vectorized memset for space padding. */
void FORTRAN(fastgetnewline, (const char *inpc, char *textpart, ITG *istat, ITG *n,
                              ITG *key, ITG *iline, ITG *ipol, ITG *inl,
                              const ITG *ipoinp, const ITG *inp, const ITG *ipoinpc)) {
  ITG il = *iline;
  ITG in = *inl;
  ITG ip = *ipol;

  if (il == inp[3 * (in - 1) + 1]) {
    if (inp[3 * (in - 1) + 2] == 0) {
      for (;;) {
        ip++;
        if (ip > NENTRIES) {
          *ipol = ip;
          *istat = -1;
          return;
        } else if (ipoinp[2 * (ip - 1)] != 0) {
          break;
        }
      }
      in = ipoinp[2 * (ip - 1)];
      il = inp[3 * (in - 1)];
    } else {
      in = inp[3 * (in - 1) + 2];
      il = inp[3 * (in - 1)];
    }
  } else {
    il++;
  }

  *iline = il;
  *inl = in;
  *ipol = ip;

  ITG start = ipoinpc[il - 1];
  ITG end = ipoinpc[il];
  ITG len = end - start;
  const char *line = &inpc[start];

  *istat = 0;
  char c0 = (len > 0) ? line[0] : ' ';
  char c1 = (len > 1) ? line[1] : ' ';
  *key = (c0 == '*' && c1 != '*') ? 1 : 0;

  ITG max_len = (len < 1319) ? len : 1319;
  ITG cur_n = 1;
  ITG j = 0;
  char *tp = textpart;

  for (ITG i = 0; i < max_len; i++) {
    char c = line[i];
    if (c != ',') {
      if (c == ' ') {
        break;
      }
      if (j < 132) {
        tp[j] = c;
      }
      j++;
    } else {
      if (j < 132) {
        memset(&tp[j], ' ', (size_t)(132 - j));
      }
      j = 0;
      cur_n++;
      tp += 132;
      if (cur_n > 16) {
        ITG ierror = 0;
        for (ITG k = i + 1; k < max_len; k++) {
          if (line[k] == ',') continue;
          if (line[k] == ' ') {
            if (ierror == 0) {
              break;
            } else {
              printf(" *ERROR in splitline: there should not\n");
              printf("        be more than 16 entries in a \n");
              printf("        line; \n");
              fwrite(line, 1, (size_t)k, stdout);
              printf("\n");
              exit(201);
            }
          }
          ierror = 1;
        }
        if (ierror != 0) {
          printf(" *ERROR in splitline: there should not\n");
          printf("        be more than 16 entries in a \n");
          printf("        line; \n");
          fwrite(line, 1, (size_t)max_len, stdout);
          printf("\n");
          exit(201);
        }
        break;
      }
    }
  }

  if (j == 0) {
    cur_n--;
  } else if (cur_n <= 16) {
    if (j < 132) {
      memset(&tp[j], ' ', (size_t)(132 - j));
    }
  }

  if (cur_n < 16) {
    memset(&textpart[cur_n * 132], ' ', (size_t)(16 - cur_n) * 132);
  }

  *n = cur_n;
}

/* Fast replacement for Fortran internal read(s(1:10), '(i10)', iostat=istat) val */
void FORTRAN(fast_read_i10, (const char *s, ITG *val, ITG *istat)) {
  int idx = 0;
  while (idx < 10 && s[idx] == ' ') idx++;
  if (idx == 10) {
    *val = 0;
    *istat = 0;
    return;
  }
  int neg = 0;
  if (s[idx] == '-') {
    neg = 1;
    idx++;
  } else if (s[idx] == '+') {
    idx++;
  }
  long long acc = 0;
  int has_digit = 0;
  for (; idx < 10; idx++) {
    unsigned char c = (unsigned char)s[idx];
    if (c >= '0' && c <= '9') {
      acc = acc * 10 + (c - '0');
      has_digit = 1;
    } else if (c == ' ') {
      continue;
    } else {
      *istat = 1;
      return;
    }
  }
  if (!has_digit) {
    *istat = 1;
    return;
  }
  *val = (ITG)(neg ? -acc : acc);
  *istat = 0;
}

static const double pow10_pos[23] = {
  1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
  1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
  1e20, 1e21, 1e22
};

/* Fast replacement for Fortran internal read(s(1:20), '(f20.0)', iostat=istat) val */
void FORTRAN(fast_read_f20, (const char *s, double *val, ITG *istat)) {
  int idx = 0;
  while (idx < 20 && s[idx] == ' ') idx++;
  if (idx == 20) {
    *val = 0.0;
    *istat = 0;
    return;
  }
  int neg = 0;
  if (s[idx] == '-') {
    neg = 1;
    idx++;
  } else if (s[idx] == '+') {
    idx++;
  }
  unsigned long long mant = 0;
  int sig_digits = 0;
  int frac_exp = 0;
  int has_digit = 0;
  int in_frac = 0;

  for (; idx < 20; idx++) {
    unsigned char c = (unsigned char)s[idx];
    if (c >= '0' && c <= '9') {
      has_digit = 1;
      if (sig_digits < 19) {
        if (mant != 0 || c != '0') {
          mant = mant * 10 + (c - '0');
          sig_digits++;
          if (in_frac) frac_exp--;
        } else if (in_frac) {
          frac_exp--;
        }
      } else {
        if (!in_frac) frac_exp++;
      }
    } else if (c == '.') {
      if (in_frac) {
        *istat = 1;
        return;
      }
      in_frac = 1;
    } else if (c == 'e' || c == 'E' || c == 'd' || c == 'D' || c == '+' || c == '-') {
      break;
    } else if (c == ' ') {
      continue;
    } else {
      *istat = 1;
      return;
    }
  }

  int exp_val = 0;
  if (idx < 20) {
    unsigned char c = (unsigned char)s[idx];
    if (c == 'e' || c == 'E' || c == 'd' || c == 'D') {
      idx++;
      while (idx < 20 && s[idx] == ' ') idx++;
    }
    int exp_neg = 0;
    if (idx < 20 && s[idx] == '-') {
      exp_neg = 1;
      idx++;
    } else if (idx < 20 && s[idx] == '+') {
      idx++;
    }
    int has_exp_digit = 0;
    for (; idx < 20; idx++) {
      unsigned char ec = (unsigned char)s[idx];
      if (ec >= '0' && ec <= '9') {
        if (exp_val < 10000) exp_val = exp_val * 10 + (ec - '0');
        has_exp_digit = 1;
      } else if (ec == ' ') {
        continue;
      } else {
        *istat = 1;
        return;
      }
    }
    if (!has_exp_digit) {
      *istat = 1;
      return;
    }
    if (exp_neg) exp_val = -exp_val;
  }

  if (!has_digit) {
    *istat = 1;
    return;
  }

  if (mant == 0) {
    *val = neg ? -0.0 : 0.0;
    *istat = 0;
    return;
  }

  int total_exp = frac_exp + exp_val;
  if (sig_digits <= 15 && total_exp >= -22 && total_exp <= 22) {
    double d = (double)mant;
    if (total_exp > 0) d *= pow10_pos[total_exp];
    else if (total_exp < 0) d /= pow10_pos[-total_exp];
    *val = neg ? -d : d;
    *istat = 0;
    return;
  }

  char buf[24];
  int bpos = 0;
  for (int k = 0; k < 20; k++) {
    char ch = s[k];
    if (ch == ' ') continue;
    if (ch == 'd' || ch == 'D') ch = 'e';
    else if ((ch == '+' || ch == '-') && bpos > 0 && buf[bpos - 1] != 'e' && buf[bpos - 1] != 'E') {
      buf[bpos++] = 'e';
    }
    buf[bpos++] = ch;
  }
  buf[bpos] = '\0';
  char *endptr = NULL;
  double d = strtod(buf, &endptr);
  if (endptr == buf || *endptr != '\0') {
    *istat = 1;
    return;
  }
  *val = d;
  *istat = 0;
}
