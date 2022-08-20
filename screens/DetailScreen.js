import React, { useEffect } from 'react';
import {View, Text, StyleSheet} from 'react-native';

function DetailScreen({route}) {
  
  return (
    <View style={styles.block}>
      <Text style={styles.text}>파워포인트 객체를 생성 중입니다.</Text>
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
});

export default DetailScreen;