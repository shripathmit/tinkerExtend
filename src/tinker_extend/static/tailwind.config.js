// Tailwind config for the tinkerExtend UI. app.css is compiled and committed,
// so the server needs no Node. Rebuild after changing index.html / app.js:
//
//   npx tailwindcss@3.4.17 -c src/tinker_extend/static/tailwind.config.js \
//     -i src/tinker_extend/static/src/input.css \
//     -o src/tinker_extend/static/app.css --minify
//
module.exports = {
  darkMode: 'class',
  content: [
    './src/tinker_extend/static/index.html',
    './src/tinker_extend/static/app.js',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        canvas: '#090A0C',
        surface: '#0E1015',
        surfaceElevated: '#13161D',
        hairline: '#181A20',
        hairlineSubtle: '#14161B',
        hairlineFocus: '#2A2E39',
        ink: '#F3F4F6',
        inkMuted: '#9CA3AF',
        inkFaint: '#5E6472',
        accent: '#3B82F6',
        amberWarn: '#F59E0B',
        emeraldOk: '#10B981',
        paperWarm: '#F4F4F0',
        paperDark: '#12151C',
      },
    },
  },
  plugins: [],
};
