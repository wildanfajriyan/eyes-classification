const {
  withNativeWind: withNativeWind
} = require("nativewind/metro");

const { getDefaultConfig } = require('expo/metro-config')

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('bin')

module.exports = withNativeWind(config, {
  input: "./global.css"
})