# Training Plate Config

This is the training-time `PlateConfig` schema validated with Pydantic. It defines plate alphabet, image
dimensions, preprocessing options, and region metadata used during training and export.

::: fast_plate_ocr.train.model.config
    options:
      filters:
        - "!^UInt8$"
