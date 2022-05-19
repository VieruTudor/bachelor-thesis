import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class AudioUploadServiceService {
  private url: string = 'http://localhost:5001/api/file/detect-emotion-from-audio';

  private results = []

  constructor(private httpClient: HttpClient) {
  }

  getResults() {
    return this.results;
  }

  detectEmotion(data: FormData){
    this.httpClient.post(this.url, data).subscribe()
  }
}


