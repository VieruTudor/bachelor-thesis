import {Component, EventEmitter, Output} from "@angular/core";
import {HttpClient} from '@angular/common/http';
import {tap} from "rxjs";
import {Prediction} from "../../models/prediction";

@Component({
  selector: 'file-picker',
  templateUrl: "file-picker.component.html",
  styleUrls: ["file-picker.component.css"]
})
export class FilePickerComponent {
  @Output() setEmotions = new EventEmitter<Prediction[]>()
  fileName = '';
  baseUrl = 'https://localhost:5001/api';

  constructor(private http: HttpClient) {
  }

  onFileSelected(event: any) {

    const file: File = event.target.files[0];

    if (file) {

      this.fileName = file.name;

      const formData = new FormData();

      formData.append("AudioFile", file);

      const upload$ = this.http.post(`${(this.baseUrl)}/file/detect-emotion-from-audio`, formData);
      upload$.pipe(
        tap((result: any) => this.setEmotions.emit(result)
        )).subscribe();
    }
  }
}
