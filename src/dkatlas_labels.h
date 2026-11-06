/***************************************************************************//**
\file    dkatlas_labels.h
@brief   brainchop DKatlas class names and colours (header-only)

By output index: 0 background, 1..34 ctx-lh-*, 35..68 ctx-rh-* (the 68
Desikan-Killiany cortical regions), then the subcortical, cerebellar and
white-matter classes. Colours are brainchop's colormap (MIT, see
src/dkatlas_LICENSE.txt). Header-only so the label writers can use it in every
build, with or without the parcellation itself.
*******************************************************************************/

#ifndef SIAMIZE_DKATLAS_LABELS_H
#define SIAMIZE_DKATLAS_LABELS_H

#include <cstdint>

namespace siam {

inline constexpr int kDKClasses = 104;

inline constexpr const char* kDKNames[kDKClasses] = {
    "BG", "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts",
    "ctx-rh-caudalanteriorcingulate", "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal",
    "ctx-rh-fusiform", "ctx-rh-inferiorparietal", "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate",
    "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal", "ctx-rh-lingual", "ctx-rh-medialorbitofrontal",
    "ctx-rh-middletemporal", "ctx-rh-parahippocampal", "ctx-rh-paracentral", "ctx-rh-parsopercularis",
    "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis", "ctx-rh-pericalcarine", "ctx-rh-postcentral",
    "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus", "ctx-rh-rostralanteriorcingulate",
    "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal", "ctx-rh-superiorparietal",
    "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole", "ctx-rh-temporalpole",
    "ctx-rh-transversetemporal", "ctx-rh-insula", "Left-Thalamus-Proper*", "Right-Thalamus-Proper*",
    "Left-Caudate", "Right-Caudate", "Left-Putamen", "Right-Putamen", "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area",
    "Right-Accumbens-area", "Left-VentralDC", "Right-VentralDC", "Left-Cerebral-White-Matter",
    "Right-Cerebral-White-Matter", "Left-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "CSF", "Brain-Stem",
    "Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter", "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior",
    "CC_Anterior"
};

inline constexpr uint8_t kDKRGB[kDKClasses][3] = {
    {0, 0, 0}, {25, 100, 40}, {125, 100, 160}, {100, 25, 0}, {220, 20, 100}, {220, 20, 10}, {180, 220, 140},
    {220, 60, 220}, {180, 40, 120}, {140, 20, 140}, {20, 30, 140}, {35, 75, 50}, {225, 140, 140},
    {200, 35, 75}, {160, 100, 50}, {20, 220, 60}, {60, 220, 60}, {220, 180, 140}, {20, 100, 50},
    {220, 60, 20}, {120, 100, 60}, {220, 20, 20}, {220, 180, 220}, {60, 20, 220}, {160, 140, 180},
    {80, 20, 140}, {75, 50, 125}, {20, 220, 160}, {20, 180, 140}, {140, 220, 220}, {80, 160, 20},
    {100, 0, 100}, {70, 70, 70}, {150, 150, 200}, {255, 192, 32}, {25, 100, 40}, {125, 100, 160},
    {100, 25, 0}, {220, 20, 100}, {220, 20, 10}, {180, 220, 140}, {220, 60, 220}, {180, 40, 120},
    {140, 20, 140}, {20, 30, 140}, {35, 75, 50}, {225, 140, 140}, {200, 35, 75}, {160, 100, 50},
    {20, 220, 60}, {60, 220, 60}, {220, 180, 140}, {20, 100, 50}, {220, 60, 20}, {120, 100, 60},
    {220, 20, 20}, {220, 180, 220}, {60, 20, 220}, {160, 140, 180}, {80, 20, 140}, {75, 50, 125},
    {20, 220, 160}, {20, 180, 140}, {140, 220, 220}, {80, 160, 20}, {100, 0, 100}, {70, 70, 70},
    {150, 150, 200}, {255, 192, 32}, {0, 118, 14}, {0, 118, 14}, {122, 186, 220}, {122, 186, 220},
    {236, 13, 176}, {236, 13, 176}, {12, 48, 255}, {13, 48, 255}, {220, 216, 20}, {220, 216, 20},
    {103, 255, 255}, {103, 255, 255}, {255, 165, 0}, {255, 165, 0}, {165, 42, 42}, {165, 42, 42},
    {245, 245, 245}, {245, 245, 245}, {120, 18, 134}, {196, 58, 250}, {120, 18, 134}, {196, 58, 250},
    {204, 182, 142}, {42, 204, 164}, {60, 60, 60}, {119, 159, 176}, {220, 248, 164}, {220, 248, 164},
    {230, 148, 34}, {230, 148, 34}, {0, 0, 64}, {0, 0, 112}, {0, 0, 160}, {0, 0, 208}, {0, 0, 255}
};

/// FreeSurfer's lookup-table code (FreeSurferColorLUT, aparc) of each cortical
/// class 1..68: ctx-lh-<region> = 1000 + its aparc index, ctx-rh-* = 2000 + it
/// (no class is FreeSurfer's 1004 / 2004, corpuscallosum); 0 background.
/// Recorded per label in the --parcel dk LabelTable (FreeSurferID); the
/// voxels keep the uint8 DK index.
inline constexpr uint16_t kDKFreeSurfer[69] = {
    0, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
    1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033,
    1034, 1035, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
    2032, 2033, 2034, 2035
};

}  // namespace siam

#endif  // SIAMIZE_DKATLAS_LABELS_H
