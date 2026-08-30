# CLE PIDR Zero-Training Gate Result

Updated: 2026-08-30

This gate reused cached softmax predictions only; no inference or training was run.
Binding and operator-family metadata were hidden during promotion-matrix estimation and opened only for scoring.

## Round 40 primary

| arm | PIDR | mAP | AUC | positive precision | positive recall | class-to-family hit |
|---|---:|---:|---:|---:|---:|---:|
| h0 | 0.024559 | 0.441855 | 0.510677 | 0.246516 | 0.468750 | 0.225000 |
| h9 | 0.175479 | 0.844847 | 0.923906 | 0.569105 | 0.906250 | 0.850000 |
| l0 | 0.025401 | 0.430622 | 0.507083 | 0.244765 | 0.431250 | 0.275000 |
| l9 | 0.174728 | 0.865557 | 0.933177 | 0.587786 | 0.925000 | 0.875000 |

```json
{
  "verdict": "GO_TO_INTERVENTION_BRIDGE_DESIGN",
  "thresholds": {
    "minimum_gamma9_map": 0.6,
    "minimum_map_delta": 0.2,
    "minimum_hit_rate": 0.7,
    "maximum_null_p": 0.01
  },
  "systems": {
    "hfl": {
      "values": {
        "gamma9_map": 0.8448474702380953,
        "map_delta": 0.40299272486772486,
        "positive_clients": 4,
        "gamma9_hit_rate": 0.85,
        "class_map_p": 0.000999000999000999,
        "probe_identity_p": 0.000999000999000999,
        "client_map_delta": [
          0.4204613095238095,
          0.4936011904761905,
          0.26466600529100526,
          0.4332423941798941
        ]
      },
      "gates": {
        "G1_gamma9_map": true,
        "G2_gamma_map_delta": true,
        "G3_client_direction": true,
        "G4_hit_rate": true,
        "G5_class_map_null": true,
        "G6_probe_identity_null": true
      },
      "pass": true
    },
    "local": {
      "values": {
        "gamma9_map": 0.8655567956349205,
        "map_delta": 0.4349351025132274,
        "positive_clients": 4,
        "gamma9_hit_rate": 0.875,
        "class_map_p": 0.000999000999000999,
        "probe_identity_p": 0.000999000999000999,
        "client_map_delta": [
          0.4183655753968254,
          0.5464657738095238,
          0.28988095238095246,
          0.4850281084656085
        ]
      },
      "gates": {
        "G1_gamma9_map": true,
        "G2_gamma_map_delta": true,
        "G3_client_direction": true,
        "G4_hit_rate": true,
        "G5_class_map_null": true,
        "G6_probe_identity_null": true
      },
      "pass": true
    }
  }
}
```

## Round 12 diagnostic

- `h0`: PIDR=0.024727, mAP=0.420916, hit=0.375000
- `h9`: PIDR=0.166220, mAP=0.813517, hit=0.800000
- `l0`: PIDR=0.024622, mAP=0.410263, hit=0.350000
- `l9`: PIDR=0.177171, mAP=0.836018, hit=0.875000

## Scope

A pass establishes only oracle-side directional observability with clean paired sources and distinguishable probes.
It does not establish that ordinary i.i.d. AugMix views overwrite an already-present degradation, nor method novelty.
