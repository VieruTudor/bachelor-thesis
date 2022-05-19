export enum Emotion {
  Neutral = 0,
  Happy = 1,
  Sad = 2,
  Angry = 3,
  Fearful = 4,
  Disgust = 5,
  Surprised = 6
}

const emotionMap = [
  {id: Emotion.Neutral, name: "Neutral"},
  {id: Emotion.Happy, name: "Happy"},
  {id: Emotion.Sad, name: "Sad"},
  {id: Emotion.Angry, name: "Angry"},
  {id: Emotion.Fearful, name: "Fearful"},
  {id: Emotion.Disgust, name: "Disgust"},
  {id: Emotion.Surprised, name: "Surprised"}
]

export const GetEmotionById = (id: Emotion) =>
  emotionMap.find(x => x.id === id)?.name;
