import React, { useEffect, useState } from 'react';
import {View, Button, StyleSheet, Text, Image, TouchableOpacity, Alert, Platform, AppRegistry} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker'; //카메라, 앨범 접근 라이브러리

function HomeScreen({navigation}) {
  const [response, setResponse] = useState(null);
  
  useEffect(() => {
    navigation.setOptions({title: 'PPT 변환기'});
  }, [navigation]);

  const onSelectImage = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        maxWidth: 512,
        maxHeight: 512,
        includeBase64: Platform.OS === 'android',
      },
      (res) => {
        if (res.didCancel) {
          //취소했을 경우
          return;
        }
        // TODO - response가 있으면 ppt로 변환하기버튼의 disabled 속성을 false로 없으면 true
        setResponse(res);
      },
    );
  };
  
  const convertPpt = async ()=>{
    console.log(response);
    
    let asset = response.assets[0];
    var photo = {
      uri: asset.uri,
      type: 'multipart/form-data', //asset.type,
      name: asset.fileName,
    }
    var body = new FormData();
    body.append('file',photo);
    
    let res = await fetch('https://2vx2xvoam5.execute-api.us-west-2.amazonaws.com/v2/uploadimage',{
      method:'POST',
      body: body,
      headers: {'content-Type': 'multipart/form-data' }
    })

    let result = await res.json();
    console.log(result);
  }
  
  return (
    <SafeAreaView>
      <View style={[styles.view]}>
        <View style={[styles.textView]}>
          <Text style={[styles.text]}>이미지 파일</Text>
          <Image
            style={[styles.imageview]}
            source={
              response
                ? {uri: response?.assets[0]?.uri}
                : require('./image_example.png')} /> 
                {/* //response 값이 존재하지 않을 경우 해당 이미지를 보여줌 */}
        </View>
    </View>
    <View style={[styles.container]}>
      <TouchableOpacity onPress={onSelectImage}>
        <View style={styles.button} backgroundColor="#2c2c2c">
          <Text style={styles.buttonText}>파일 불러오기</Text>
        </View>
      </TouchableOpacity>
      <TouchableOpacity
        // onPress={() => {
        //   if(response == null){
        //     Alert.alert('이미지가 없습니다.');
        //     console.log('오류 메시지 나타남');
        //     return;
        //   }
        //   else {
        //     navigation.navigate('Detail');
        //     convertPpt();
        //   }
        // }}>
        onPress={() => {navigation.navigate('Detail');}}>
          <View style={styles.button}>
            <Text style={styles.buttonText}>PPT로 변환하기</Text>
          </View>
        </TouchableOpacity>
    </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  text: {fontSize: 20},
  textView: {width: '100%', padding: 10, marginBottom: 10},
  view: {justifyContent: 'space-between', alignItems: 'center'},
  textInput: {fontSize: 24, padding: 10, borderWidth: 1, borderRadius: 5},
  container: {
    // paddingTop: 5,
    alignItems: 'center'
  },
  buttonText: {
    textAlign: 'center',
    padding: 10,
    color: 'white'
  },
  imageview:{
    height: 200,
    width: 400,
    transform: [{scale: 0.7}],
    alignItems: 'center'
  },
  button: {
    width: 260,
    height: 60,
    alignItems: 'center',
    backgroundColor: '#2196F3',
    padding: 10,
    marginBottom: 30,
    borderRadius: 80
  },
})

export default HomeScreen;