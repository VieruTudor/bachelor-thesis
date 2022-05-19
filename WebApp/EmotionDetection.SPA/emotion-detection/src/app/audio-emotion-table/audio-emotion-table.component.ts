import {Component, Input, OnInit} from '@angular/core';
import {Prediction} from "../../models/prediction";
import {Emotion, GetEmotionById} from "../constants/emotions";

@Component({
  selector: 'audio-emotion-table',
  templateUrl: './audio-emotion-table.component.html',
  styleUrls: ['./audio-emotion-table.component.css']
})
export class AudioEmotionTableComponent implements OnInit {

  _predictions: Prediction[] = []

  get getPredictions(): Prediction[] {
    return this._predictions;
  }

  @Input() set setPredictions(predictions: Prediction[]) {
    this._predictions = predictions;
    this.dataSource = this._predictions;
  }

  displayedColumns = ['emotion', 'value'];
  getEmotionById = GetEmotionById
  dataSource: Prediction[] = [{emotion: Emotion.Neutral, level: 100.0}]

  constructor() {

  }

  ngOnInit(): void {
  }

}
