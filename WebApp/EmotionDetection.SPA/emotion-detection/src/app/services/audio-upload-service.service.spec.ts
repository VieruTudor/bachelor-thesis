import { TestBed } from '@angular/core/testing';

import { AudioUploadServiceService } from './audio-upload-service.service';

describe('AudioUploadServiceService', () => {
  let service: AudioUploadServiceService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AudioUploadServiceService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
