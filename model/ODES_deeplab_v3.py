import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm
from .aspp import build_aspp



class seg_model(nn.Module):
    def __init__(self, num_class):
        super(seg_model, self).__init__()
        self.num_class = num_class
        self.encoder = timm.create_model('resnet18', pretrained= True, features_only=True, out_indices=[1,4])
    
        low_level_inplanes = 512 #for resnet18
        aspp_outplanes = 256
        self.aspp = build_aspp(inplanes = low_level_inplanes, outplanes = aspp_outplanes)
        
        #decoder
        self.conv1 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, self.num_class, kernel_size=1, stride=1))
        
            

 
    def forward(self, inp, require_feature=False):

        features = self.encoder(inp)
        low_level_feat, x = features
        x = self.aspp(x)
        
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        concat_features = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(concat_features)
        output = F.interpolate(x, size=inp.size()[2:], mode='bilinear', align_corners=True)

        if require_feature:
            return output, features, concat_features
        else:
            return output











# class SegmentationModel(nn.Module):
#     def __init__(self, num_classes):
#         super(SegmentationModel, self).__init__()
#         self.num_classes = num_classes

#         # Encoder: Pretrained ResNet18 with feature extraction
#         self.encoder = timm.create_model('resnet18', pretrained=True, features_only=True)

#         # ASPP Module for high-level features
#         self.aspp = build_aspp(inplanes=512, outplanes=256)

#         # Low-level feature processing
#         self.low_level_processing = nn.Sequential(
#             nn.Conv2d(64, 48, kernel_size=1, bias=False),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )

#         # Decoder for feature refinement
#         self.decoder = nn.Sequential(
#             nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1)
#         )

#         # Final classification layer
#         self.classifier = nn.Conv2d(256, self.num_classes, kernel_size=1)

#     def forward(self, x, require_feature=False):
#         """
#         Forward pass of the segmentation model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, C, H, W).
#             require_feature (bool): Whether to return intermediate features.

#         Returns:
#             torch.Tensor or tuple: Segmentation map or tuple with intermediate features.
#         """
#         # Extract features from the encoder
#         features = self.encoder(x)
#         low_level_feat, high_level_feat = features[1], features[4]

#         # Process high-level features with ASPP
#         high_level_feat = self.aspp(high_level_feat)

#         # Process low-level features
#         low_level_feat = self.low_level_processing(low_level_feat)

#         # Upsample and concatenate features
#         high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
#         concat_features = torch.cat([high_level_feat, low_level_feat], dim=1)

#         # Decode features
#         decoded_features = self.decoder(concat_features)

#         # Generate segmentation output
#         output = self.classifier(decoded_features)
#         output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

#         if require_feature:
#             return output, features, concat_features, decoded_features
#         return output


# if __name__ == "__main__":
#     # Test case with dummy input
#     batch_size, channels, height, width = 2, 3, 256, 256
#     dummy_input = torch.randn(batch_size, channels, height, width)
#     num_classes = 5

#     model = SegmentationModel(num_classes=num_classes)
#     output, features, concat_features, decoded_features = model(dummy_input, require_feature=True)

#     print("--- Shape Summary ---")
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Low-level feature shape (features[1]): {features[1].shape}")
#     print(f"High-level feature shape (features[4]): {features[4].shape}")
#     print(f"Concatenated feature shape: {concat_features.shape}")
#     print(f"Decoded feature shape: {decoded_features.shape}")
#     print(f"Output shape: {output.shape}")
