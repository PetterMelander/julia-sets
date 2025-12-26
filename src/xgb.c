#include "xgb.h"

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
  if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.43772488832)) {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.022982208058)) {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)-0.12751688063)) {
        result[0] += 0.47134092;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016595913097)) {
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
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.10217755288)) {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.62898743153)) {
            result[0] += 0.40494594;
          } else {
            result[0] += -0.581772;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.00091236067237)) {
            result[0] += 0.27550286;
          } else {
            result[0] += -0.801425;
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
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.10610286146)) {
            result[0] += 0.39210626;
          } else {
            result[0] += 0.23514755;
          }
        } else {
          result[0] += -0.15657;
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016595913097)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.25934896;
          } else {
            result[0] += -0.39835143;
          }
        } else {
          result[0] += 0.40715265;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          result[0] += -0.524833;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.4053716;
          } else {
            result[0] += -0.34340516;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.73109775782)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.010034662671)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.88249063492)) {
          result[0] += -0.48051897;
        } else {
          if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.62631630898)) {
            result[0] += 0.28550413;
          } else {
            result[0] += -0.09265476;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.048252608627)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.042388193309)) {
            result[0] += -0.49784318;
          } else {
            result[0] += 0.32392517;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)1.032145977)) {
            result[0] += 0.3349393;
          } else {
            result[0] += -0.5204758;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42544105649)) {
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
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016595913097)) {
        result[0] += 0.36245126;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021919997409)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.086584553123)) {
            result[0] += -0.43646348;
          } else {
            result[0] += 0.32212245;
          }
        } else {
          result[0] += 0.33434907;
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.73109775782)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.011911300942)) {
        if ( (data[1].missing != -1) && (data[1].fvalue < (float)-0.012817583978)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-1.6661295891)) {
            result[0] += 0.19161557;
          } else {
            result[0] += 0.3834853;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)1.0019619465)) {
            result[0] += 0.11807368;
          } else {
            result[0] += -0.4919635;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.048252608627)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.072880581021)) {
            result[0] += -0.42689928;
          } else {
            result[0] += 0.2884473;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.7296654582)) {
            result[0] += -0.13632056;
          } else {
            result[0] += 0.37570307;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42544105649)) {
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
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016595913097)) {
        result[0] += 0.34044898;
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.086584553123)) {
            result[0] += -0.42435843;
          } else {
            result[0] += 0.25444794;
          }
        } else {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)-1.7592526674)) {
            result[0] += 0.09528948;
          } else {
            result[0] += 0.32806492;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.81492471695)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.010454600677)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.028362224;
          } else {
            result[0] += -0.55390316;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)1.0019619465)) {
            result[0] += 0.29763234;
          } else {
            result[0] += -0.07775172;
          }
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.048252608627)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.92054241896)) {
            result[0] += -0.11753186;
          } else {
            result[0] += -0.36641514;
          }
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.29208326;
          } else {
            result[0] += -0.38490382;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.42544105649)) {
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
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.017861522734)) {
          result[0] += 0.31956083;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
            result[0] += -0.35521144;
          } else {
            result[0] += 0.27461845;
          }
        }
      }
    } else {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.016595913097)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.63530516624)) {
          if ( (data[1].missing != -1) && (data[1].fvalue < (float)0.80267548561)) {
            result[0] += 0.18385254;
          } else {
            result[0] += -0.272145;
          }
        } else {
          result[0] += 0.3525709;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.021498618647)) {
          result[0] += -0.34532437;
        } else {
          if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.71074926853)) {
            result[0] += 0.3522975;
          } else {
            result[0] += -0.24961819;
          }
        }
      }
    }
  } else {
    if ( (data[4].missing != -1) && (data[4].fvalue < (float)0.91971057653)) {
      if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.078826531768)) {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)-0.39772447944)) {
          if ( (data[0].missing != -1) && (data[0].fvalue < (float)-0.088228985667)) {
            result[0] += 0.018034717;
          } else {
            result[0] += -0.57130605;
          }
        } else {
          result[0] += 0.36871386;
        }
      } else {
        if ( (data[2].missing != -1) && (data[2].fvalue < (float)0.069415211678)) {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.15663450956)) {
            result[0] += 0.20760308;
          } else {
            result[0] += -0.3232473;
          }
        } else {
          if ( (data[3].missing != -1) && (data[3].fvalue < (float)-0.7296654582)) {
            result[0] += -0.15578051;
          } else {
            result[0] += 0.3671033;
          }
        }
      }
    } else {
      if ( (data[4].missing != -1) && (data[4].fvalue < (float)1.0454556942)) {
        if ( (data[3].missing != -1) && (data[3].fvalue < (float)0.34679269791)) {
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
