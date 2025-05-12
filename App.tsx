import React, { useEffect, useRef, useState } from "react";
import { View, StyleSheet, TouchableOpacity, Image, Text, Button, Pressable } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";
import * as jpeg from "jpeg-js";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { defaultConfig } from '@tamagui/config/v4'
import {CameraView, CameraType, useCameraPermissions, Camera, CameraMode } from "expo-camera";
import { Spinner } from "./components/ui/spinner";


export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState("");
  const [pickedImage, setPickedImage] = useState("");
  // const [facing, setFacing] = useState<CameraType>("back");
  // const [permission, requestPermission] = useCameraPermissions();
  // const cameraRef = useRef(null);

  const [permission, requestPermission] = useCameraPermissions();
  const ref = useRef<CameraView>(null);
  const [uri, setUri] = useState<string | null>(null);
  const [mode, setMode] = useState<CameraMode>("picture");
  const [facing, setFacing] = useState<CameraType>("back");
  const [recording, setRecording] = useState(false);
  const [showCamera, setShowCamera] = useState(false);

  useEffect(() => {
    classify();
  }, [pickedImage]);

  const getTensor = async () => {
    const modelWeightOne = await require("./group1-shard1of2.bin");
    const modelWeightTwo = await require("./group1-shard2of2.bin");
    const modelJson = await require("./model.json");
    const model = await tf.loadLayersModel(
      bundleResourceIO(modelJson, [modelWeightOne, modelWeightTwo])
    );
    return model;
  };

  // This function helps you call the image picker and enable the user to choose an image for classification
  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
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
      setResult("");
      // Load mobilenet
      await tf.ready();
      // const model = await mobilenet.load();
      const model = await getTensor();
      setIsTfReady(true);
      console.log("starting inference with picked image: " + pickedImage);

      // Convert image to tensor
      const imgB64 = await FileSystem.readAsStringAsync(pickedImage, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
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
        setResult(
          prediction.toFixed(4)
        );
      }

      console.log("pred", JSON.stringify(prediction));
      // return prediction.toString();
    } catch (err) {
      console.log(err);
    }
  };

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  if (!permission) {
    return null;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: "center" }}>
          We need your permission to use the camera
        </Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const takePicture = async () => {
    const photo = await ref.current?.takePictureAsync();
    setPickedImage(photo?.uri)
    setShowCamera(false);
    // setUri(photo?.uri);
  };

  const toggleMode = () => {
    setMode((prev) => (prev === "picture" ? "video" : "picture"));
  };

  const toggleFacing = () => {
    setFacing((prev) => (prev === "back" ? "front" : "back"));
  };

  const closeCamera = () => {
    setShowCamera(false);
    // Optionally pause preview
    if (ref.current) {
      ref.current?.pausePreview(); // or cameraRef.current.stopRecording();
    }
  };

  return (
    <View
      style={{
        height: '100%',
        backgroundColor: 'red',
        display: 'flex',
        flexDirection: 'column',
        // alignItems: 'center',
        // justifyContent: 'center',
      }}
    >

      {showCamera && (
 <CameraView
 style={styles.camera}
 ref={ref}
 mode={mode}
 facing={facing}
 mute={false}
 responsiveOrientationWhenOrientationLocked
>
 <View style={styles.shutterContainer}>
 <Button title="Close Camera" onPress={closeCamera} />

   <Pressable>
     {mode === "picture" ? (
       // <AntDesign name="picture" size={32} color="white" />
       <Text>Picture</Text>
     ) : (
       // <Feather name="video" size={32} color="white" />
       <></>
     )}
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
               backgroundColor: mode === "picture" ? "white" : "red",
             },
           ]}
         />
       </View>
     )}
   </Pressable>
   <Pressable onPress={toggleFacing}>
     {/* <FontAwesome6 name="rotate-left" size={32} color="white" /> */}
     <Text>rotate</Text>
   </Pressable>
 </View>

</CameraView>
      )}
     

      <Image
        source={{ uri: pickedImage }}
        style={{ width: 200, height: 200, margin: 40 }}
      />

      {/* <Text>{uri}</Text> */}

      {isTfReady && <Button onPress={pickImage} title="Upload Image" />}
      <Button title="Open Camera" onPress={() => setShowCamera(true)} />

      {!isTfReady && <Spinner size="large" />}

      {result !== '' || !pickedImage ? (
        <Text>
          {Number(result) === 0
            ? ''
            : Number(result) > 0.5
            ? 'Normal'
            : 'Cataract'}
        </Text>
      ) : (
        <Spinner size="small" />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    flex: 1,
    width: "100%",
  },
  shutterContainer: {
    position: "absolute",
    bottom: 44,
    left: 0,
    width: "100%",
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 30,
  },
  shutterBtn: {
    backgroundColor: "transparent",
    borderWidth: 5,
    borderColor: "white",
    width: 85,
    height: 85,
    borderRadius: 45,
    alignItems: "center",
    justifyContent: "center",
  },
  shutterBtnInner: {
    width: 70,
    height: 70,
    borderRadius: 50,
  },
});