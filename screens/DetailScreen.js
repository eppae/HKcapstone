import React, { useEffect, useState } from 'react';
import {View, Text, StyleSheet, TouchableOpacity, Alert} from 'react-native';
import RNFetchBlob from 'rn-fetch-blob';

function DetailScreen({route}) {
  const [response] = useState(null);

  const downloadppt = async () => {
    try{
      await RNFetchBlob.config({
        addAndroidDownloads: {
          useDownloadManager: true,
          notification: true,
          path: `${RNFetchBlob.fs.dirs.DocumentDir}/${file.name}`,
          description: 'Downloading the file',
        },
      })
      config(options).fetch('GET','URL');
    } catch(error){
      console.log('fail download');
    }
  };

  return (
    <View style={styles.block}>
      <TouchableOpacity
        onPress={() => {
          downloadppt();
          // if(response == null){
          //   console.log('오류 메시지');
          //   Alert.alert('변환에 실패하였습니다.');
          //   return;
          // }
        }}>
        <View style={styles.button}>
          <Text style={styles.buttonText}>PPT 저장하기</Text>
        </View>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  block: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 28,
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
  buttonText: {
    textAlign: 'center',
    padding: 10,
    color: 'white'
  },
});

export default DetailScreen;