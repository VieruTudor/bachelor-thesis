import { Component } from '@angular/core';
import {Prediction} from "../models/prediction";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'emotion-detection';
  emotions : Prediction[] = []

  setEmotions(emotions: Prediction[]){
    this.emotions = emotions;
    console.table(this.emotions)
  }
}



