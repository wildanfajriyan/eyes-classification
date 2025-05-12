import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Image,
  Text,
  Button,
  Pressable,
} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as jpeg from 'jpeg-js';
import * as mobilenet from '@tensorflow-models/mobilenet';
import {
  CameraView,
  CameraType,
  useCameraPermissions,
  Camera,
  CameraMode,
} from 'expo-camera';

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState('');
  const [pickedImage, setPickedImage] = useState('');
  const [permission, requestPermission] = useCameraPermissions();
  const ref = useRef<CameraView>(null);
  const [mode, setMode] = useState<CameraMode>('picture');
  const [facing, setFacing] = useState<CameraType>('back');
  const [showCamera, setShowCamera] = useState(false);

  useEffect(() => {
    classify();
  }, [pickedImage]);

  const getTensor = async () => {
    const modelWeightOne = await require('./group1-shard1of2.bin');
    const modelWeightTwo = await require('./group1-shard2of2.bin');
    const modelJson = await require('./model.json');
    const model = await tf.loadLayersModel(
      bundleResourceIO(modelJson, [modelWeightOne, modelWeightTwo])
    );
    return model;
  };

  // This function helps you call the image picker and enable the user to choose an image for classification
  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setPickedImage(result.assets[0].uri);
    }
  };

  const makePredictions = async (
    model: tf.LayersModel,
    img: tf.Tensor<tf.Rank>
  ) => {
    const prediction = model.predict(img) as tf.Tensor;
    const values = prediction.dataSync()[0];
    return values;
  };

  const classify = async () => {
    try {
      setResult('');
      // Load mobilenet
      await tf.ready();
      // const model = await mobilenet.load();
      const model = await getTensor();
      setIsTfReady(true);
      console.log('starting inference with picked image: ' + pickedImage);

      // Convert image to tensor
      const imgB64 = await FileSystem.readAsStringAsync(pickedImage, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);

      const imageTensor = decodeJpeg(raw);

      const scalar = tf.scalar(255);

      const tensorScaled = imageTensor.div(scalar);
      const grayscale = tf.mean(tensorScaled, 2).expandDims(2);

      const resized = tf.image.resizeBilinear(grayscale, [150, 150]);

      const img = tf.reshape(resized, [1, 150, 150, 1]);

      // Classify the tensor and show the result
      const prediction = await makePredictions(model, img);

      if (prediction) {
        setResult(prediction.toFixed(4));
      }

      console.log('pred', JSON.stringify(prediction));
      // return prediction.toString();
    } catch (err) {
      console.log(err);
    }
  };

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }

  if (!permission) {
    return null;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>
          We need your permission to use the camera
        </Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const takePicture = async () => {
    const photo = await ref.current?.takePictureAsync();
    setPickedImage(photo?.uri);
    setShowCamera(false);
    // setUri(photo?.uri);
  };

  const toggleFacing = () => {
    setFacing((prev) => (prev === 'back' ? 'front' : 'back'));
  };

  const closeCamera = () => {
    setShowCamera(false);

    if (ref.current) {
      ref.current?.pausePreview(); // or cameraRef.current.stopRecording();
    }
  };

  if (showCamera) {
    return (
      <CameraView
        style={styles.camera}
        ref={ref}
        mode={mode}
        facing={facing}
        mute={false}
        ratio='4:3'
        responsiveOrientationWhenOrientationLocked
      >
        <View style={styles.shutterContainer}>
        {/* <Button title="Close Camera" onPress={closeCamera} /> */}
        <Pressable onPress={closeCamera}>
            {/* <FontAwesome6 name="rotate-left" size={32} color="white" /> */}
            <Text style={{color: 'white'}}>close</Text>
          </Pressable>

          <Pressable onPress={takePicture}>
            {({ pressed }) => (
              <View
                style={[
                  styles.shutterBtn,
                  {
                    opacity: pressed ? 0.5 : 1,
                  },
                ]}
              >
                <View
                  style={[
                    styles.shutterBtnInner,
                    {
                      backgroundColor: mode === 'picture' ? 'white' : 'red',
                    },
                  ]}
                />
              </View>
            )}
          </Pressable>
          <Pressable onPress={toggleFacing}>
            {/* <FontAwesome6 name="rotate-left" size={32} color="white" /> */}
            <Text style={{color: 'white'}}>rotate</Text>
          </Pressable>
        </View>
      </CameraView>
    );
  }

  return (
    <View
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Image
        source={{ uri: pickedImage }}
        style={{ width: 200, height: 200, margin: 40 }}
      />

      <View style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <View style={{ display: 'flex', flexDirection: 'row', gap: 10 }}>
          {isTfReady && <Button onPress={pickImage} title="Upload Image" />}
          <Button title="Open Camera" onPress={() => setShowCamera(true)} />
        </View>

        {!isTfReady && <Text>LOADING...</Text>}

        {result !== '' || !pickedImage ? (
          <Text style={{ fontWeight: 'bold' }}>
            {Number(result) === 0
              ? ''
              : Number(result) > 0.5
              ? 'Normal'
              : 'Cataract'}
          </Text>
        ) : (
          <Text>LOADING...</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#fff'
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  shutterContainer: {
    position: 'absolute',
    bottom: 70,
    left: 0,
    width: '100%',
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 30,
  },
  shutterBtn: {
    backgroundColor: 'transparent',
    borderWidth: 5,
    borderColor: 'white',
    width: 85,
    height: 85,
    borderRadius: 45,
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
  },
  shutterBtnInner: {
    width: 70,
    height: 70,
    borderRadius: 50,
  },
});
