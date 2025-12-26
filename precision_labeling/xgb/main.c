
#include "header.h"


static const int32_t num_class[] = {  1, };

int32_t get_num_target(void) {
  return N_TARGET;
}
void get_num_class(int32_t* out) {
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t get_num_feature(void) {
  return 5;
}
const char* get_threshold_type(void) {
  return "float32";
}
const char* get_leaf_output_type(void) {
  return "float32";
}

void predict(union Entry* data, int pred_margin, float* result) {
  unsigned int tmp;
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.43772488832)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.022982208058)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.12751688063)) {
        result[0] += 0.47134092;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016663005576)) {
          result[0] += 0.45930284;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021919997409)) {
            result[0] += -0.5588599;
          } else {
            result[0] += 0.4311219;
          }
        }
      }
    } else {
      if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.086584553123)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.10179171711)) {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.62699347734)) {
            result[0] += 0.40105116;
          } else {
            result[0] += -0.581772;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.00091236067237)) {
            result[0] += 0.27550286;
          } else {
            result[0] += -0.8013821;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += 0.15016627;
          } else {
            result[0] += -0.6535744;
          }
        } else {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.53397840261)) {
            result[0] += 0.46556294;
          } else {
            result[0] += 0.28773856;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
      if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.032602135092)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.15663450956)) {
          if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.81492471695)) {
            result[0] += 0.24753135;
          } else {
            result[0] += -0.6151605;
          }
        } else {
          result[0] += -0.80382866;
        }
      } else {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.93024301529)) {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
            result[0] += -0.10322377;
          } else {
            result[0] += 0.45994785;
          }
        } else {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += -0.54993886;
          } else {
            result[0] += 0.33080834;
          }
        }
      }
    } else {
      result[0] += -0.8151166;
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.16539770365)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.10662018508)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.1692917645)) {
        result[0] += 0.40711263;
      } else {
        if ( (data[1].missing != -1) && (data[1].fvalue < (float)1.087911725)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.10529461503)) {
            result[0] += 0.39210626;
          } else {
            result[0] += 0.23514755;
          }
        } else {
          result[0] += -0.15657;
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016663005576)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.25941825;
          } else {
            result[0] += -0.39835143;
          }
        } else {
          result[0] += 0.40727475;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          result[0] += -0.5248412;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.40553877;
          } else {
            result[0] += -0.34340516;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.73109775782)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.0099867302924)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.88249063492)) {
          result[0] += -0.47923344;
        } else {
          if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.62631630898)) {
            result[0] += 0.2856213;
          } else {
            result[0] += -0.09265476;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.041261736304)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.043298453093)) {
            result[0] += -0.50091374;
          } else {
            result[0] += 0.32921773;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)1.032145977)) {
            result[0] += 0.33545843;
          } else {
            result[0] += -0.5204758;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42597350478)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.85899162292)) {
            result[0] += 0.28665403;
          } else {
            result[0] += -0.49224412;
          }
        } else {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += -0.43975487;
          } else {
            result[0] += 0.3904421;
          }
        }
      } else {
        result[0] += -0.526479;
      }
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.022982208058)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.12751688063)) {
      result[0] += 0.36929965;
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016663005576)) {
        result[0] += 0.36244613;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021919997409)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.086584553123)) {
            result[0] += -0.43646243;
          } else {
            result[0] += 0.32212245;
          }
        } else {
          result[0] += 0.33433872;
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.73109775782)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.01094801724)) {
        if ( (data[1].missing != -1) && (data[1].fvalue < (float)-0.012817583978)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-1.6661295891)) {
            result[0] += 0.19174318;
          } else {
            result[0] += 0.38305518;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)1.0019619465)) {
            result[0] += 0.1211972;
          } else {
            result[0] += -0.4920073;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.041261736304)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.073922611773)) {
            result[0] += -0.42386332;
          } else {
            result[0] += 0.28785822;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.7296654582)) {
            result[0] += -0.13582604;
          } else {
            result[0] += 0.3758045;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42597350478)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)1.3038954735)) {
            result[0] += -0.4003422;
          } else {
            result[0] += 0.061744224;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.93024301529)) {
            result[0] += 0.36387265;
          } else {
            result[0] += -0.3183815;
          }
        }
      } else {
        result[0] += -0.43214345;
      }
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.022982208058)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.1692917645)) {
      result[0] += 0.34815302;
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016663005576)) {
        result[0] += 0.34044653;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.086584553123)) {
            result[0] += -0.42435798;
          } else {
            result[0] += 0.25444794;
          }
        } else {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)-1.7592526674)) {
            result[0] += 0.09528551;
          } else {
            result[0] += 0.3280615;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.81492471695)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.010454600677)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.027792294;
          } else {
            result[0] += -0.55474454;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)1.0019619465)) {
            result[0] += 0.29746997;
          } else {
            result[0] += -0.07772271;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.041261736304)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.92054241896)) {
            result[0] += -0.10980415;
          } else {
            result[0] += -0.36847633;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.2923628;
          } else {
            result[0] += -0.38501784;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42597350478)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-1.0313020945)) {
            result[0] += 0.31190163;
          } else {
            result[0] += -0.3802761;
          }
        } else {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += -0.30391726;
          } else {
            result[0] += 0.3344001;
          }
        }
      } else {
        result[0] += -0.3838465;
      }
    }
  }
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.18633981049)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.10662018508)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.19016885757)) {
        result[0] += 0.3337196;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.017937375233)) {
          result[0] += 0.31956103;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
            result[0] += -0.35521156;
          } else {
            result[0] += 0.27461886;
          }
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016663005576)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.1836534;
          } else {
            result[0] += -0.27243072;
          }
        } else {
          result[0] += 0.35387948;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          result[0] += -0.34542644;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.34797242;
          } else {
            result[0] += -0.24967752;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.91971057653)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.078201279044)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.39772447944)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += 0.017620413;
          } else {
            result[0] += -0.5716372;
          }
        } else {
          result[0] += 0.36864913;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.069415211678)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.15663450956)) {
            result[0] += 0.20763339;
          } else {
            result[0] += -0.3233749;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.7296654582)) {
            result[0] += -0.15555248;
          } else {
            result[0] += 0.36710823;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.3473328054)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-1.0313020945)) {
            result[0] += 0.0137607595;
          } else {
            result[0] += -0.3437463;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.76087635756)) {
            result[0] += 0.28362602;
          } else {
            result[0] += -0.21540251;
          }
        }
      } else {
        result[0] += -0.35473728;
      }
    }
  }
  
  // Apply base_scores
  result[0] += 0.5485408460001841568;
  
  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void postprocess(float* result) {
  // sigmoid
  const float alpha = (float)1;
  for (size_t i = 0; i < N_TARGET * MAX_N_CLASS; ++i) {
    result[i] = (float)(1) / ((float)(1) + expf(-alpha * result[i]));
  }
}

